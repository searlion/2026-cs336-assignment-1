import regex as re
from collections import Counter
from multiprocessing import Pool
from os import cpu_count
from functools import partial
from typing import BinaryIO
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


type frequency = int
type byte_pair = tuple[bytes, bytes]
type encoded_word = list[bytes] # A list bytes, where each bytes is a pre-tokenized word
type encoded_word_hashable = bytes # bytes, where each byte is a pre-tokenized word
type PairCounts = dict[byte_pair, frequency] # A dictionary mapping a byte pair to its frequency    
type input_path = str
type start = int
type end = int

# Pre-tokenization
def tokenizing_chunk(chunk):
    return re.finditer(PAT, chunk)

# # Encode the pre-tokenized words as bytes
def encode_as_hashable_bytes(chunk) -> encoded_word_hashable:
    return chunk.encode("utf-8")

def encode_as_bytes(bytestring: bytes) -> encoded_word:
    return [bytes([b]) for b in bytestring]

def pairs_from_encoded_word(word: encoded_word) -> PairCounts:
    return Counter(zip(word[:-1], word[1:]))

def calculate_lexicographically_largest_byte_pair_with_highest_frequency(straight_index: dict[byte_pair, frequency]) -> tuple[byte_pair, frequency]:
    max_frequency_pair = max(straight_index.items(), key=lambda item: (item[1], item[0]))
    return max_frequency_pair[0], max_frequency_pair[1]

def count_pretokens(chunk_spec: tuple[input_path, start, end], special_tokens: list[str]) -> dict[encoded_word_hashable, frequency]:
    input_path, start, end = chunk_spec
    word_id_count : dict[encoded_word_hashable, frequency] = {}
    escape_special_tokens = "|".join([re.escape(token) for token in special_tokens])
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        segments = re.split(escape_special_tokens, chunk)
        for segment in segments:
            for match in tokenizing_chunk(segment):
                encoded_hashable = encode_as_hashable_bytes(match.group())
                if encoded_hashable not in word_id_count.keys():
                    word_id_count[encoded_hashable] = 1
                else:
                    word_id_count[encoded_hashable] += 1
    return word_id_count

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


# if __name__ == "__main__":
def train_bpe(input_path: str, vocab_size: int = 500, special_tokens: list[str] = []) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = cpu_count()

    ## initialization of data structures for counting byte pairs and word pairs
    word_id_set : dict[encoded_word_hashable, encoded_word] = {} # Each unique pre-token gets an integer id.
    word_id_count : dict[encoded_word_hashable, frequency] = {} # How many times that word occured (its frequency weight)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    specs = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as p:
        count_pretokens_with_special_tokens = partial(count_pretokens, special_tokens=special_tokens)
        for chunk_word_id_count in p.imap_unordered(count_pretokens_with_special_tokens, specs):
            for word_id_this, frequency_this in chunk_word_id_count.items():
                if word_id_this not in word_id_set.keys():
                    word_id_set[word_id_this] = encode_as_bytes(word_id_this)
                    word_id_count[word_id_this] = frequency_this
                else:
                    word_id_count[word_id_this] += frequency_this

    ### BELOW CODE ARE NON-PARALLELISABLE  
    byte_pair_index : dict[byte_pair, set[encoded_word_hashable]] = {} # an inverted index telling you where each pair lives, so after a merge you only touch affected words
    straight_index : dict[byte_pair, frequency] = {} # The thing you argmax over to pick the next merge.

    ## Below code builds the byte pair index, straight index, and reverse index for the pre-tokenized words
    for word_id_this in word_id_set.keys():
        word_id_this_frequency = word_id_count[word_id_this]
        word_encoded_this = word_id_set[word_id_this]
        counters = pairs_from_encoded_word(word_encoded_this)
        for tuple_pair, frequency_count in counters.items():
            straight_index[tuple_pair] = straight_index.get(tuple_pair, 0) + frequency_count * word_id_this_frequency
            byte_pair_index.setdefault(tuple_pair, set()).add(word_id_this)

    vocab : dict[int, bytes] = {i: bytes([i]) for i in range(256)} # The vocab is a mapping from the integer id to the bytes representation of the pre-tokenized word
    for i in range(256, 256 + len(special_tokens)):
        vocab[i] = special_tokens[i - 256].encode("utf-8") # Add the special tokens to the vocab
    merges : list[tuple[bytes, bytes]] = [] # The merges is a list of tuples of the merged tokens, in the order they were merged

    while len(vocab) < vocab_size and straight_index:
        ## Get the lexicographically largest byte pair with the highest frequency
        max_frequency_pair, max_frequency = calculate_lexicographically_largest_byte_pair_with_highest_frequency(straight_index)
        vocab_index = len(vocab)
        vocab[vocab_index] = max_frequency_pair[0] + max_frequency_pair[1]
        merges.append(max_frequency_pair)

        # Merge tokens
        # Wrapped byte_pair_index in a list to avoid "Set changed size during iteration" error
        for word_id_this in list(byte_pair_index[max_frequency_pair]):  
            # Update the word_id_set with the merged tokens
            word_index = 0
            word_encoded_this = word_id_set[word_id_this]
            word_encoded_length = len(word_encoded_this)
            new_word_encoded_this = []
            while word_index < word_encoded_length:
                # Append the last token if it wasn't merged
                if word_index == word_encoded_length - 1:
                    new_word_encoded_this.append(word_encoded_this[word_index])
                    break
                if (word_encoded_this[word_index], word_encoded_this[word_index + 1]) == max_frequency_pair:
                    # Merge the pair
                    merged_token = word_encoded_this[word_index] + word_encoded_this[word_index + 1]
                    new_word_encoded_this.append(merged_token)
                    word_index += 2
                else:
                    new_word_encoded_this.append(word_encoded_this[word_index])
                    word_index += 1
            old_pairs = pairs_from_encoded_word(word_encoded_this)
            word_id_set[word_id_this] = new_word_encoded_this
            new_pairs = pairs_from_encoded_word(new_word_encoded_this)
            word_freq = word_id_count[word_id_this]
            # Update the straight_index with the new pairs
            for delta_pairs in old_pairs.keys() | new_pairs.keys():
                change = (new_pairs.get(delta_pairs, 0) - old_pairs.get(delta_pairs, 0)) * word_freq
                straight_index[delta_pairs] = straight_index.get(delta_pairs, 0) + change
                assert straight_index[delta_pairs] >= 0, f"Straight index for {delta_pairs} is negative: {straight_index[delta_pairs]}"
                if straight_index[delta_pairs] == 0:
                    del straight_index[delta_pairs] 
                if change > 0:
                    byte_pair_index.setdefault(delta_pairs, set()).add(word_id_this)
                elif change < 0:
                    if new_pairs.get(delta_pairs, 0) == 0:
                        byte_pair_index[delta_pairs].remove(word_id_this)
                    if len(byte_pair_index[delta_pairs]) == 0:
                        del byte_pair_index[delta_pairs]

    return vocab, merges