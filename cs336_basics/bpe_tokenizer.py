from typing import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
            self.special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        else:
            self.special_pattern = None

        self.reverse_vocab = {}
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k

    # Copied and modified from bpe_trainer.py
    def pretokenize_chunk(self, chunk: str) -> list[tuple[bytes]]:
        pre_tokens = []
        if self.special_pattern is not None:
            mini_chunks = re.split(f"({self.special_pattern})", chunk)
        else:
            mini_chunks = [chunk]
        for mini_chunk in mini_chunks:
            if self.special_tokens is not None and mini_chunk in self.special_tokens:
                pre_tokens.append(tuple([mini_chunk.encode("utf-8")]))
            else:
                rex = re.finditer(PAT, mini_chunk)
                
                for match in rex:
                    pre_token = match.group().encode("utf-8")
                    pre_tokens.append(tuple(bytes([b]) for b in pre_token))
        return pre_tokens

    def from_files(self):
        pass

    def encode(self, text: str) -> list[int]:
        output_token_ids = []
        pts = self.pretokenize_chunk(text)
        for pt in pts:
            merged_pt = self.merge_pt(pt)
            for token in merged_pt:
                output_token_ids.append(self.reverse_vocab[token])
        return output_token_ids

    def merge_pt(self, pt):
        def get_all_pairs(pt):
            all_pairs = set()

            for pair in zip(pt[:-1], pt[1:]):
                all_pairs.add(pair)
            return all_pairs
        
        all_pairs = get_all_pairs(pt)
        for merge_pair in self.merges:
            if merge_pair in all_pairs:
                new_pt = []
                i = 0
                while i < len(pt) - 1:
                    if pt[i] == merge_pair[0] and pt[i+1] == merge_pair[1]:
                        new_pt.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_pt.append(pt[i])
                        i += 1
                    if i == len(pt) - 1:
                        new_pt.append(pt[i])
                pt = tuple(new_pt)
                all_pairs = get_all_pairs(pt)
        return pt
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token in self.encode(chunk):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        output_bytes = b''
        for id in ids:
            output_bytes += self.vocab[id]
        return output_bytes.decode("utf-8", errors="replace")
    