from datetime import datetime
from pathlib import Path
import pickle

import regex as re
import os
from collections import Counter
import itertools
from tqdm import tqdm

PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 





def generate_initial_mapping(special_tokens):
    mapping = {i : int.to_bytes(i) for i in range(256)}
    mapping.update({256 + i: special_token.encode('utf-8') for i, special_token in enumerate(special_tokens)})
    return mapping



def do_merges(max_num_merges: int, pretoken_counts: dict[bytes, int]) -> list[tuple[int, int]]:
    vocab = [int.to_bytes(i, 1) for i in range(256)]
    merges = []
    pretoken_counts = [(''.join(chr(x) for x in k), v) for k,v in pretoken_counts.items()]

    two_gram_pretoken_index = {}

    two_gram_count = Counter()
    for i, (pretoken, count) in enumerate(pretoken_counts):
        for a, b in zip(pretoken, pretoken[1:]):
            two_gram_count[(a,b)] += count

            pretoken_list = two_gram_pretoken_index.setdefault((a,b), set())
            pretoken_list.add(i)


    for _ in tqdm(range(max_num_merges), desc="Performing merges"):
        two_gram_to_merge = max(two_gram_count.items(), key=lambda x: (x[1], vocab[ord(x[0][0])], vocab[ord(x[0][1])]))[0]

        two_gram_count.pop(two_gram_to_merge)
        a, b = two_gram_to_merge
        
        merges.append((vocab[ord(a)], vocab[ord(b)]))
        vocab.append(vocab[ord(a)] + vocab[ord(b)])
        new_token =  chr(len(vocab) - 1)
        for pretoken_ind in two_gram_pretoken_index[(a,b)]:
            pretoken, count = pretoken_counts[pretoken_ind]
            if a+b not in pretoken:
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

                    two_gram_count[(left,right)] -= count
                    two_gram_count[(x,y)] += count
                    pretoken_list = two_gram_pretoken_index.setdefault((x,y), set())
                    pretoken_list.add(pretoken_ind)
        two_gram_pretoken_index.pop((a,b))

    return vocab, merges





def split_by_special_token(data_str: str, special_tokens: list[str]):
    special_tokens_pattern = '|'.join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return re.split(special_tokens_pattern, data_str)



def sum_counters_fast(counters):
    counters = sorted(counters, key=len, reverse=True)
    total = counters[0].copy()
    for c in counters[1:]:
        total.update(c)
    return total


def pre_tokenize(data_chunks: list[str], special_tokens: list[str]) -> dict[bytes, int]:
    pretoken_count = Counter()
    for data_str in tqdm(data_chunks, desc="Pretokenizing chunks"):
        pretoken_count.update(x.group() for x in re.finditer(PRETOKENIZATION_PATTERN, data_str))
    pretoken_count_dict = dict(pretoken_count)
    return {k.encode() : v for k,v in pretoken_count_dict.items() if k not in special_tokens}






def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    corpus = Path.read_text(input_path, encoding='utf-8')
    print("read corpus")
    # corpus = corpus.decode()
    data_chunks = split_by_special_token(corpus, special_tokens)
    pretoken_counts = pre_tokenize(data_chunks, special_tokens)
    vocab, merges = do_merges(vocab_size - 256 - len(special_tokens), pretoken_counts)
    vocab.extend(token.encode() for token in special_tokens)
    return dict(enumerate(vocab)), merges




def train_bpe(data_file_name: str, vocab_size: int, special_tokens: list[str]):
    data_file_path = (Path(__file__).parent.parent / "data" / data_file_name).with_suffix('.txt')
    out_file_name = f"{data_file_name}-bpe-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    out_file_path = Path(__file__).parent.parent / "models" / out_file_name

    output = run_train_bpe(data_file_path, vocab_size, special_tokens)
    with open(out_file_path, 'wb') as f:
        pickle.dump(output, f)
    return output



Class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        special_tokens = special_tokens or list()

    @classmethod
    















FIXTURES_PATH = (Path(__file__).resolve().parent.parent) / "tests" / "fixtures"

if __name__ == '__main__':
    vocab, merges = train_bpe(
        "TinyStoriesV2-GPT4-train",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )


