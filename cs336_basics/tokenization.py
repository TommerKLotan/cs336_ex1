import itertools
import os
import pickle
import struct
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import regex as re
from tqdm import tqdm

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAIN_DIR = Path(__file__).parent.parent
FIXTURES_PATH = MAIN_DIR / "tests" / "fixtures"
TOKENIZATION_MODELS_DIR = MAIN_DIR / "tokenization_models"
DATA_DIR = MAIN_DIR / "data"
TOKENIZED_DATA_DIR = MAIN_DIR / "tokenized_data"


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


def get_pretoken_counts(data_chunks: Iterable[str]) -> dict[bytes, int]:
    pretoken_count = Counter()
    for data_str in tqdm(data_chunks, desc="Pretokenizing chunks"):
        pretoken_count.update(pretokenize(data_str))
    pretoken_count_dict = dict(pretoken_count)
    return {k.encode(): v for k, v in pretoken_count_dict.items()}


def train_chunk_generator(input_path: str, special_tokens: list[str]):
    BUFFER_SIZE = 100_000
    buffer = ""
    with open(input_path, encoding="utf-8") as f:
        for str_chunk in f:
            buffer += str_chunk
            if len(buffer) >= BUFFER_SIZE:
                chunks = split_by_special_token(buffer, special_tokens)
                buffer = chunks.pop()
                yield from chunks
    chunks = split_by_special_token(buffer, special_tokens)
    yield from chunks


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    data_chunks = train_chunk_generator(input_path, special_tokens)
    pretoken_counts = get_pretoken_counts(data_chunks)
    vocab, merges = create_merges(vocab_size - 256 - len(special_tokens), pretoken_counts)
    vocab.extend(token.encode() for token in special_tokens)
    return dict(enumerate(vocab)), merges


def train_bpe_to_file(data_file_name: str, vocab_size: int, special_tokens: list[str]):
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
    while len(tokens) > 1:
        pairs = zip(tokens, tokens[1:])
        merge_candidates = [pair for pair in pairs if pair in merge_rank_dict]
        if not merge_candidates:
            break
        a, b = min(merge_candidates, key=merge_rank_dict.get)
        tokens = replace_subarray(tokens, [a, b], [a + b])
    return [inverse_vocab[x] for x in tokens]


def encode_pretokens(pretokens: list[str], inverse_vocab, merges) -> list[int]:
    merge_rank_dict = {m: i for i, m in enumerate(merges)}
    return list(itertools.chain(*(encode_pretoken(pretoken, inverse_vocab, merge_rank_dict) for pretoken in pretokens)))


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
        encoded = list(
            itertools.chain(
                *(
                    [self.inverse_vocab[chunk.encode()]]
                    if chunk in self.special_tokens
                    else encode_pretokens(pretokenize(chunk), self.inverse_vocab, self.merges)
                    for chunk in tqdm(chunks)
                )
            )
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
                    buffer = "".join([*pretokens[-2:], buffer[-max_special_token_len:]])
                    yield from encode_pretokens(pretokens[:-2], self.inverse_vocab, self.merges)
                    # print(total)

            if buffer:
                yield from self.encode(buffer)
                return

        return token_id_generator()

    def decode(self, ids: list[int]) -> str:
        print(ids)
        token_bytes = b"".join(self.vocab[id] for id in ids)
        return token_bytes.decode(errors="replace")


def tokenize_text_to_file(tokenizer_file_name: str, data_file_name: str, special_tokens=None):

    tokenizer = Tokenizer.from_files(TOKENIZATION_MODELS_DIR / tokenizer_file_name, special_tokens)
    with open(DATA_DIR / data_file_name) as f:
        ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save(TOKENIZED_DATA_DIR / (f"{tokenizer_file_name}-{data_file_name}"), ids)


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe_to_file(
    #     "TinyStoriesV2-GPT4-train",
    #     vocab_size=10000,
    #     special_tokens=special_tokens,
    # )

    # Train on OWT
    # vocab, merges = train_bpe_to_file(
    #     "owt_train.txt",
    #     vocab_size=32000,
    #     special_tokens=special_tokens,
    # )

    # Tokenize tiny stories
    for data_file_name in (f"TinyStoriesV2-GPT4-{x}.txt" for x in ["train", "valid"]):
        tokenize_text_to_file(
            "tinystories_sample_5M-bpe-2026-10-07_19-15-50", data_file_name, special_tokens=special_tokens
        )

    # print("aaaaa")
    # tokenizer = Tokenizer.from_files(MODELS_DIR / "TinyStoriesV2-GPT4-train-bpe-2026-10-08_11-29-06", special_tokens=[])
    # with open(DATA_DIR / "tinystories_sample_5M.txt") as f:
    #     print("starting")
    #     for x in zip(range(100000), tokenizer.encode_iterable(f)):
    #         pass
    # encoded = tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")
    # print(encoded)
