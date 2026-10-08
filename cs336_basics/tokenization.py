import os
import pickle
import struct
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import regex as re
from tqdm import tqdm

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def replace_subarray(lst, target, replacement):
    if not target:
        return lst

    target_len = len(target)
    lst_len = len(lst)
    result = []

    i = 0
    while i < lst_len:
        # Check if the target sequence matches without slicing the original list
        if i <= lst_len - target_len and lst[i : i + target_len] == target:
            result.extend(replacement)
            i += target_len
        else:
            result.append(lst[i])
            i += 1

    return result


def create_merges(max_num_merges: int, pretoken_counts: dict[bytes, int]) -> list[tuple[int, int]]:
    vocab = [int.to_bytes(i, 1) for i in range(256)]
    merges = []
    pretoken_counts = [("".join(chr(x) for x in k), v) for k, v in pretoken_counts.items()]

    two_gram_pretoken_index = {}

    two_gram_count = Counter()
    for i, (pretoken, count) in enumerate(pretoken_counts):
        for a, b in zip(pretoken, pretoken[1:]):
            two_gram_count[(a, b)] += count

            pretoken_list = two_gram_pretoken_index.setdefault((a, b), set())
            pretoken_list.add(i)

    for _ in tqdm(range(max_num_merges), desc="Performing merges"):
        two_gram_to_merge = max(two_gram_count.items(), key=lambda x: (x[1], vocab[ord(x[0][0])], vocab[ord(x[0][1])]))[
            0
        ]

        two_gram_count.pop(two_gram_to_merge)
        a, b = two_gram_to_merge

        merges.append((vocab[ord(a)], vocab[ord(b)]))
        vocab.append(vocab[ord(a)] + vocab[ord(b)])
        new_token = chr(len(vocab) - 1)
        for pretoken_ind in two_gram_pretoken_index[(a, b)]:
            pretoken, count = pretoken_counts[pretoken_ind]
            if a + b not in pretoken:
                continue
            new_pretoken = pretoken.replace(a + b, new_token)
            pretoken_counts[pretoken_ind] = (new_pretoken, count)
            for x, y in zip(new_pretoken, new_pretoken[1:]):
                if x == new_token or y == new_token:
                    if x == new_token:
                        left = b
                    else:
                        left = x
                    if y == new_token:
                        right = a
                    else:
                        right = y

                    two_gram_count[(left, right)] -= count
                    two_gram_count[(x, y)] += count
                    pretoken_list = two_gram_pretoken_index.setdefault((x, y), set())
                    pretoken_list.add(pretoken_ind)
        two_gram_pretoken_index.pop((a, b))

    return vocab, merges


def split_by_special_token(data_str: str, special_tokens: list[str], keep_special_tokens=False):
    if not special_tokens:
        return [data_str]
    special_tokens_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    pattern = f"({special_tokens_pattern})" if keep_special_tokens else special_tokens_pattern
    return [x for x in re.split(pattern, data_str) if x]


def sum_counters_fast(counters):
    counters = sorted(counters, key=len, reverse=True)
    total = counters[0].copy()
    for c in counters[1:]:
        total.update(c)
    return total


def pretokenize(data: str) -> Iterator[str]:
    return (x.group() for x in re.finditer(PRETOKENIZATION_PATTERN, data))


def get_pretoken_counts(data_chunks: list[str], special_tokens: list[str]) -> dict[bytes, int]:
    pretoken_count = Counter()
    for data_str in tqdm(data_chunks, desc="Pretokenizing chunks"):
        pretoken_count.update(pretokenize(data_str))
    pretoken_count_dict = dict(pretoken_count)
    return {k.encode(): v for k, v in pretoken_count_dict.items() if k not in special_tokens}


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    corpus = Path.read_text(input_path, encoding="utf-8")
    data_chunks = split_by_special_token(corpus, special_tokens)
    pretoken_counts = get_pretoken_counts(data_chunks, special_tokens)
    vocab, merges = create_merges(vocab_size - 256 - len(special_tokens), pretoken_counts)
    vocab.extend(token.encode() for token in special_tokens)
    return dict(enumerate(vocab)), merges


def train_bpe(data_file_name: str, vocab_size: int, special_tokens: list[str]):
    data_file_path = (Path(__file__).parent.parent / "data" / data_file_name).with_suffix(".txt")
    out_file_name = f"{data_file_name}-bpe-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_file_path = Path(__file__).parent.parent / "models" / out_file_name

    output = run_train_bpe(data_file_path, vocab_size, special_tokens)
    with open(out_file_path, "wb") as f:
        pickle.dump(output, f)
    return output


def encode_pretoken(pretoken: str, inverse_vocab, merge_rank_dict):

    encoded_pretoken = pretoken.encode()
    tokens = list(x[0] for x in struct.iter_unpack("c", encoded_pretoken))
    pairs = zip(tokens, tokens[1:])
    merge_candidates = [pair for pair in pairs if pair in merge_rank_dict]
    sorted_merge_candidates = sorted()

    for a, b in merges:
        if len(tokens) < 2:
            break
        if a+b in encoded_pretoken:
            tokens = replace_subarray(tokens, [a, b], [a + b])
    return [inverse_vocab[x] for x in tokens]


def encode_pretokens(pretokens: list[str], inverse_vocab, merges) -> list[int]:
    merge_rank_dict = {m: i for i, m in enumerate(merges)}
    return sum((encode_pretoken(pretoken, inverse_vocab, merge_rank_dict) for pretoken in pretokens), [])

    


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        special_tokens = special_tokens or list()
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_and_merges_filepath: str, special_tokens=None):
        with open(vocab_and_merges_filepath, "rb") as f:
            vocab, merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special_token(text, self.special_tokens, keep_special_tokens=True)
        ()
        encoded = sum(
            (
                [self.inverse_vocab[chunk.encode()]]
                if chunk in self.special_tokens
                else encode_pretokens(pretokenize(chunk), self.inverse_vocab, self.merges)
                for chunk in tqdm(chunks)
            ),
            [],
        )
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        BUFFER_SIZE = 50000
        max_special_token_len = max((len(x) for x in self.special_tokens), default=1)
        def token_id_generator():
            total = 0
            buffer = ""
            for string in iterable:
                buffer = buffer + string
                total += len(string)
                special_token = next((x for x in self.special_tokens if x in buffer), None)
                if special_token is not None:
                    chunk, buffer = buffer.split(special_token, maxsplit=1)
                    pretokens = pretokenize(chunk)
                    yield from encode_pretokens(pretokens, self.inverse_vocab, self.merges)
                    yield self.inverse_vocab[special_token.encode()]
                elif len(buffer) > BUFFER_SIZE:
                    pretokens = list(pretokenize(buffer[:-max_special_token_len]))
                    buffer = ''.join([*pretokens[-2:], buffer[-max_special_token_len:]])
                    yield from encode_pretokens(pretokens[:-2], self.inverse_vocab, self.merges)
                    # print(total)
                    
            if buffer:
                print('here', len(buffer))
                yield from self.encode(buffer)
                print('here now')
                return

        return token_id_generator()

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[id] for id in ids)
        return token_bytes.decode(errors="replace")


FIXTURES_PATH = (Path(__file__).resolve().parent.parent) / "tests" / "fixtures"
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe(
    #     "TinyStoriesV2-GPT4-train",
    #     vocab_size=10000,
    #     special_tokens=special_tokens,
    # )
    print('aaaaa')
    tokenizer = Tokenizer.from_files(MODELS_DIR / "TinyStoriesV2-GPT4-train-bpe-2026-10-08_11-29-06", special_tokens=[])
    with open(DATA_DIR / 'tinystories_sample_5M.txt') as f:
        print('starting')
        for x in zip(range(100000), tokenizer.encode_iterable(f)):
            pass
    # encoded = tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")

