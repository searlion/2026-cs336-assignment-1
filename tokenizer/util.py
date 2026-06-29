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
    byte_pair_index : dict[byte_pair, list[tuple[word_id, word_offset]]] = {} # an inverted index telling you where each pair lives, so after a merge you only touch affected words
    straight_index : dict[byte_pair, frequency] = {} # The thing you argmax over to pick the next merge.
    reverse_index : dict[frequency, set[byte_pair]] = {} # Meant to let you grab the current max-frequency pair quickly
    word_id_set : set[list[bytes]] = {} # Each unique pre-token gets an integer id.
    word_set = set() # Dedup helper: have I seen this word before?
    word_id_count : dict[word_id, frequency] = {} # How many times that word occured (its frequency weight)

    ## Below code assigns a word_id to each pre-tokenized word, and counts its frequency
    for match in tokenizing_chunk(chunk):
        encoded = encode_as_bytes(match.group())
        if encoded not in word_set:
            word_set.add(encoded)
            word_id_this = len(word_id_set)
            word_id_set.add(encoded)
            word_id_count[word_id_this] = 1
        else:
            word_id_this = word_id_set[encoded]
            word_id_count[word_id_this] += 1
    
    for word_id_this in word_id_set:
        word_id_this_frequency = word_id_count[word_id_this]
        offset_count = 0
        for tuple_pair in zip(word_id_this[:-1], word_id_this[1:]):
            tuple_pair = (bytes([tuple_pair[0]]), bytes([tuple_pair[1]])) # to convert (int, int) to (bytes, bytes)
            straight_index[tuple_pair] = straight_index.get(tuple_pair, 0) + word_id_this_frequency
            byte_pair_index[tuple_pair].append((word_id_this, offset_count))
            offset_count += 1
        for pair, count in straight_index.items():
            if count not in reverse_index:
                reverse_index[count] = [pair]
            else:
                reverse_index[count].append(pair)