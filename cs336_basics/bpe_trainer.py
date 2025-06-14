import os
import time
import pickle
import cProfile
import multiprocessing
from multiprocessing import Pool
from dataclasses import dataclass
from typing import BinaryIO
from abc import ABC
from collections import defaultdict, Counter
import regex as re

NUM_PROCESSES = multiprocessing.cpu_count() - 1

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Copied from the example
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_bytes_pair(pair):
    pair = bytes([pair[0]]), bytes([pair[1]])
    return pair

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: list[tuple[bytes, bytes]]

class BPETrainer:
    def __init__(self, special_tokens):
        self.special_token = special_tokens[0]
        self.special_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
        self.vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
        for token in special_tokens:
            token_idx = len(self.vocab)
            self.vocab[token_idx] = token.encode("utf-8")
        self.merges: list[tuple[bytes, bytes]] = []
    
    def decode_pair(self, merge_pair):
        t1, t2 = merge_pair
        merged_bytes = t1 + t2
        return merged_bytes

    def pretokenize_chunk(self, chunk: str) -> dict[tuple[bytes], int]:
        mini_chunks = self.special_pattern.split(chunk)
        rexs = [re.finditer(PAT, mini_chunk) for mini_chunk in mini_chunks]
        pre_tokens = []
        for rex in rexs:
            for match in rex:
                pre_token = match.group().encode("utf-8")
                pre_tokens.append(tuple(bytes([b]) for b in pre_token))
        counter_pre_tokens = Counter(pre_tokens)
        return counter_pre_tokens
    
    def process_chunk(self, args):
        fname, start, end = args
        with open(fname, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            counter_pt = self.pretokenize_chunk(chunk)
        return counter_pt

    def process_chunks_for_pretokenization(self, fname, boundaries):
        self.counter_pt = Counter()

        args_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args_list.append((fname, start, end))
        
        with Pool(NUM_PROCESSES) as pool:
            for counter_pt in pool.imap_unordered(self.process_chunk, args_list):
                self.counter_pt.update(counter_pt)

    def update(self):
        merge_pair = max(self.counter_bp_fq.items(), key=lambda x: (x[1], x[0]))[0]
        v = self.counter_bp_fq[merge_pair]
        self.counter_bp_fq.pop(merge_pair)

        updated_pts = self.map_pair_pt[merge_pair]
        del self.map_pair_pt[merge_pair]
        for k in updated_pts:
            v = self.counter_pt[k]
            for pair in zip(k[:-1], k[1:]):
                self.counter_bp_fq.subtract({pair: v})
                if self.counter_bp_fq[pair] == 0:
                    del self.counter_bp_fq[pair]
        
        token_idx = len(self.vocab)
        self.vocab[token_idx] = self.decode_pair(merge_pair)

        self.merges.append(merge_pair)
        merged_pair = self.vocab[token_idx]

        for k in updated_pts:
            before_k = list(k)
            after_k = []

            i = 0
            while i < len(k) - 1:
                if k[i] == merge_pair[0] and k[i+1] == merge_pair[1]:
                    after_k.append(merged_pair)
                    i += 2
                else:
                    after_k.append(k[i])
                    i += 1
            if i == len(k) - 1:
                after_k.append(k[i])
            after_k = tuple(after_k)

            v = self.counter_pt[k]
            del self.counter_pt[k]

            self.counter_pt[after_k] = v
            if len(after_k) == 1: continue
            for pair in zip(after_k[:-1], after_k[1:]):
                self.counter_bp_fq.update({pair: v})
                self.map_pair_pt[pair].add(after_k)

    def train_bpe(self, fname, vocab_size):
        boundaries = self.get_chunk_boundaries(fname)

        self.process_chunks_for_pretokenization(fname, boundaries)

        # now create byte pairs and their frequencies
        self.counter_bp_fq = Counter()
        self.map_pair_pt = defaultdict(set)

        i = 0
        for k, v in self.counter_pt.items():
            i += 1
            if len(k) == 1: continue
            for pair in zip(k[:-1], k[1:]):
                self.counter_bp_fq.update({pair: v})
                self.map_pair_pt[pair].add(k)

        total_updates = vocab_size - len(self.vocab)
        for _ in range(total_updates):
            self.update()
        
        return BPETokenizerParams(vocab=self.vocab, merges=self.merges)
    
    def get_chunk_boundaries(self, fname):
        with open(fname, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8"))

        print(f"chunk boundaries: {boundaries}")
        return boundaries

def save_params(tokenizer_params: BPETokenizerParams, fname):
    with open(f"{fname}.pickle", "wb") as f:
        pickle.dump(tokenizer_params, f)

def analyze_cprofile(fname):
    import pstats
    from pstats import SortKey
    p = pstats.Stats(fname)
    
    print("Total time taken for my functions")
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('bpe_trainer.py')
    
## Function to be called for testing
def run_train_bpe(
        input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
        save_to_disk="",
    ):
    bpe_trainer = BPETrainer(special_tokens=special_tokens)
    
    tokenizer_params = bpe_trainer.train_bpe(input_path, vocab_size)
    if len(save_to_disk) > 0:
        save_params(tokenizer_params, save_to_disk)

    return tokenizer_params.vocab, tokenizer_params.merges

def train_bpe_tinystories():
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    tic = time.time()
    run_train_bpe(input_path, vocab_size=10000, save_to_disk="tinystories")
    print(f"Took {time.time() - tic} sec ... ")

def train_bpe_expts_owt():
    input_path = "../data/owt_train.txt"
    tic = time.time()
    run_train_bpe(input_path, vocab_size=32000, save_to_disk="owt")
    print(f"Took {time.time() - tic} sec ... ")

def test_small():
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    tic = time.time()
    run_train_bpe(input_path, vocab_size=500, save_to_disk="")
    print(f"Took {time.time() - tic} sec ... ")

if __name__ == "__main__":
    # cProfile.run('test_small()', 'test_small')
    # analyze_cprofile('test_small')
    cProfile.run('train_bpe_tinystories()', 'profile_tinystories')
    analyze_cprofile('profile_tinystories')





# ================================
# Stats for tinystories

# Took 601.1359770298004 sec ...
# Total time taken for my functions
# Sat Jun  7 01:04:10 2025    profile_tinystories

#          2607126095 function calls (2607125713 primitive calls) in 601.137 seconds

#    Ordered by: cumulative time
#    List reduced from 625 to 12 due to restriction <'bpe_trainer.py'>

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    1.937    1.937  601.137  601.137 bpe_trainer.py:224(train_bpe_tinystories)
#         1    0.000    0.000  599.200  599.200 bpe_trainer.py:210(run_train_bpe)
#         1    0.240    0.240  599.198  599.198 bpe_trainer.py:166(train_bpe)
#      9743   95.564    0.010  404.957    0.042 bpe_trainer.py:121(update)
#         1    0.000    0.000  193.625  193.625 bpe_trainer.py:110(process_chunks_for_pretokenization)
# 874823485   30.937    0.000   30.937    0.000 bpe_trainer.py:122(<lambda>)
#         1    0.000    0.000    0.002    0.002 bpe_trainer.py:197(save_params)
#      9743    0.001    0.000    0.001    0.000 bpe_trainer.py:86(decode_pair)
#         1    0.000    0.000    0.001    0.001 bpe_trainer.py:189(get_chunk_boundaries)
#         1    0.000    0.000    0.001    0.001 bpe_trainer.py:18(find_chunk_boundaries)
#         1    0.000    0.000    0.000    0.000 bpe_trainer.py:77(__init__)
#         2    0.000    0.000    0.000    0.000 bpe_trainer.py:79(<genexpr>)

# Took 10 minutes on Macbook Air M3
# update() took 400 sec, which may be because i'm iterating over the counter_bp_fq dict
# instead of optimizing that.
