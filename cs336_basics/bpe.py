import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

def pretokenization_single(content: bytes, split_special_token: bytes) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    sub_chunks = content.split(split_special_token)
    result = []
    for sub_chunk in sub_chunks:
        s = sub_chunk.decode("utf-8")
        result.extend(re.findall(PAT, s))
    return result

def pretokenization(input_path: str | os.PathLike, num_processes = 1, split_special_token=b"<|endoftext|>") -> list[str]:
    with open(input_path, "rb") as f:
        content = f.read()
        if num_processes <= 1:
            return pretokenization_single(content, split_special_token)
        else:
            with open(input_path, "rb") as f:
                boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
                if len(boundaries) == 2:
                    return pretokenization_single(content, split_special_token)
                chunks = []
                for bi in range(len(boundaries) - 1):
                    f.seek(boundaries[bi])
                    chunks.append(f.read(boundaries[bi + 1] - boundaries[bi]))
                with Pool(processes=len(chunks)) as pool:
                    chunk_results = pool.starmap(pretokenization_single, [(c, split_special_token) for c in chunks])
                    results = []
                    for chunk_result in chunk_results:
                        results.extend(chunk_result)
                    return results

def sort_by_value(d):
    return sorted(d.items(), key=lambda p: p[1], reverse=True)

class BPE:
    def __init__(self, num_merges, special_tokens: list[str]):
        self.num_merges = num_merges
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

        for i in range(256):
            self.vocab[len(self.vocab)] = bytes([i])
        for special_token in special_tokens:
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")

    def train(self, pretokenized_words: list[str]):
        sequences: list[list[int]] = []
        for word in pretokenized_words:
            encoded = word.encode("utf-8")
            sequences.append([encoded[i] for i in range(len(encoded))])

        pair_counts: dict[tuple[int], int] = {} # byte-pair: count
        pair_in_sequences: dict[tuple[int], set[int]] = {} # byte-pair: set(index), to speed-up updated
        for idx, sequence in enumerate(sequences):
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i+1])
                if pair not in pair_counts:
                    pair_counts[pair] = 1
                    pair_in_sequences[pair] = set([idx])
                else:
                    pair_counts[pair] += 1
                    pair_in_sequences[pair].add(idx)

        merges = self.merges
        vocab = self.vocab
        for round in range(self.num_merges):
            # merge candidate which counts most, when equal
            candidates = []
            max_count = 0
            for pair, count in pair_counts.items():
                if count > max_count:
                    candidates = [pair]
                    max_count = count
                elif count == max_count:
                    candidates.append(pair)
            if not candidates:
                break
            # pick lexico-max candidate
            picked = candidates[0]
            for i in range(len(candidates)):
                candidate = candidates[i]
                if candidate > picked:
                    picked = candidate
            
            # update vocab & merges
            merges.append((vocab[picked[0]], vocab[picked[1]]))
            new_token = len(vocab)
            self.vocab[new_token] = vocab[picked[0]] + vocab[picked[1]]
    
            # check affected sequences
            affected_sequence_indices = list(pair_in_sequences[picked])
            for idx in affected_sequence_indices:
                sequence = sequences[idx]
                new_sequence = []
    
                # cound old pairs
                old_pair_counts: dict[tuple[int], int] = {} # byte-pair: count
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i + 1])
                    if pair not in old_pair_counts:
                        old_pair_counts[pair] = 1
                    else:
                        old_pair_counts[pair] += 1
    
                i = 0
                # merge sequence
                while i < len(sequence) - 1:
                    if (sequence[i], sequence[i + 1]) == picked:
                        new_sequence.append(new_token)
                        i += 2
                    else:
                        new_sequence.append(sequence[i])
                        i += 1
                if i == len(sequence) - 1:
                    new_sequence.append(sequence[i])
                sequences[idx] = new_sequence
    
                # count new pairs
                new_pair_counts: dict[tuple[int], int] = {} # byte-pair: count
                for i in range(len(new_sequence) - 1):
                    pair = (new_sequence[i], new_sequence[i + 1])
                    if pair not in new_pair_counts:
                        new_pair_counts[pair] = 1
                    else:
                        new_pair_counts[pair] += 1
            
                for pair, count in old_pair_counts.items():
                    pair_counts[pair] -= count
                    if pair not in new_pair_counts:
                        pair_in_sequences[pair].remove(idx)
                for pair, count in new_pair_counts.items():
                    if pair not in pair_counts:
                        pair_counts[pair] = count
                        pair_in_sequences[pair] = set([idx])
                    else:
                        pair_counts[pair] += count
                        pair_in_sequences[pair].add(idx)

    def encode(self, text: str) -> list[int]:
        byte_str = text.encode("utf-8")
        tokens = list(byte_str)

        # must keep merge order
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, tokens: list[str]) -> str:
        byte_str = b''.join([self.vocab[token] for token in tokens])
        return byte_str.decode("utf-8")


def run_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Byte-pair encoding (BPE) algorithm implementation

    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pretokenized_words = pretokenization(input_path, 1)

    num_merges = vocab_size - 256 - len(special_tokens)
    bpe = BPE(num_merges, special_tokens)

    bpe.train(pretokenized_words)

    return bpe.vocab, bpe.merges

FIXTURES_PATH = "./tests/fixtures"
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json

def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())

if __name__ == "__main__":
    input_path = FIXTURES_PATH / "corpus.en"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 500

    num_merges = vocab_size - 256 - len(special_tokens)
    bpe = BPE(num_merges, special_tokens)
    pretokenized_words = pretokenization(input_path, 1)
    bpe.train(pretokenized_words)

    with open("decoded.txt", "wb") as f:
        for word in pretokenized_words:
            f.write(bpe.decode(bpe.encode(word)).encode("utf-8"))
