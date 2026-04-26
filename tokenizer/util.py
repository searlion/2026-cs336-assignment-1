import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

type word_id = int
type word_offset = int
type frequency = int
type byte_pair = tuple[bytes, bytes]

def tokenizing_chunk(chunk):
    # This is a placeholder function. You should replace it with your actual tokenization logic.
    return re.finditer(PAT, chunk)

def encode_as_bytes(chunk):
    # This is a placeholder function. You should replace it with your actual encoding logic.
    return chunk.encode("utf-8")


if __name__ == "__main__":
    chunk = "some text that i'll pre-tokenize"
    ## initialization of data structures for counting byte pairs and word pairs
    byte_pair_index : dict[byte_pair, tuple[frequency, list[tuple[word_id, word_offset]]]] = {}
    reverse_index : dict[frequency, list[byte_pair]] = {}
    word_id_list : list[list[bytes]] = []
    word_set = set()
    word_pair_count : dict[word_id, frequency] = {}
    count : dict[tuple[bytes, ...], int] = {}
    for match in tokenizing_chunk(chunk):
        encoded = encode_as_bytes(match.group())
        if encoded not in word_set:
            word_set.add(encoded)
            word_id = len(word_id_list)
            word_id_list.append(encoded)
            word_pair_count[word_id] = 1
        else:
            word_id = word_id_list.index(encoded)
            word_pair_count[word_id] += 1

        for tuple_pair in zip(encoded[:-1], encoded[1:]):
            count[tuple_pair] = count.get(tuple_pair, 0) + 1
    print(count)