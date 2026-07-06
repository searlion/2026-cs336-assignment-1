import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


type word_id = int 
type word_offset = int
type frequency = int
type byte_pair = tuple[bytes, bytes]
type encoded_word = list[list[bytes]] # A list of lists of bytes, where each list of bytes is a pre-tokenized word

# Pre-tokenization
def tokenizing_chunk(chunk):
    return re.finditer(PAT, chunk)

# # Encode the pre-tokenized words as bytes
# def encode_as_bytes(chunk):
#     return chunk.encode("utf-8")

def encode_as_bytes(bytestring: bytes):
    return [bytes([b]) for b in bytestring]

if __name__ == "__main__":
    chunk = "some text that i'll pre-tokenize"
    ## initialization of data structures for counting byte pairs and word pairs
    byte_pair_index : dict[byte_pair, set[word_id]] = {} # an inverted index telling you where each pair lives, so after a merge you only touch affected words
    straight_index : dict[byte_pair, frequency] = {} # The thing you argmax over to pick the next merge.
    reverse_index : dict[frequency, set[byte_pair]] = {} # Meant to let you grab the current max-frequency pair quickly
    word_id_set : dict[word_id, encoded_word] = {} # Each unique pre-token gets an integer id.
    word_id_set_reverse : dict[encoded_word, word_id] = {} # Reverse lookup for the above
    word_set = set() # Dedup helper: have I seen this word before?
    word_id_count : dict[word_id, frequency] = {} # How many times that word occured (its frequency weight)

    ## Below code assigns a word_id to each pre-tokenized word, and counts its frequency
    for match in tokenizing_chunk(chunk):
        encoded = encode_as_bytes(match.group().encode("utf-8"))
        if encoded not in word_set:
            word_set.add(encoded)
            word_id_this = len(word_id_set)
            word_id_set[word_id_this] = encoded
            word_id_set_reverse[encoded] = word_id_this
            word_id_count[word_id_this] = 1
        else:
            word_id_this = word_id_set_reverse[encoded]
            word_id_count[word_id_this] += 1
    
    ## Below code builds the byte pair index, straight index, and reverse index for the pre-tokenized words
    for word_id_this in word_id_set.keys():
        word_id_this_frequency = word_id_count[word_id_this]
        word_encoded_this = word_id_set[word_id_this]
        for tuple_pair in zip(word_encoded_this[:-1], word_encoded_this[1:]):
            tuple_pair = (bytes([tuple_pair[0]]), bytes([tuple_pair[1]])) # to convert (int, int) to (bytes, bytes)
            straight_index[tuple_pair] = straight_index.get(tuple_pair, 0) + word_id_this_frequency
            byte_pair_index[tuple_pair].add(word_id_this)
        for pair, count in straight_index.items():
            if count not in reverse_index:
                reverse_index[count] = [pair]
            else:
                reverse_index[count].append(pair)

    ## Get the lexicographically largest byte pair with the highest frequency
    max_frequency = max(reverse_index.keys())
    max_frequency_pairs = reverse_index[max_frequency]
    max_frequency_pair = max(max_frequency_pairs, key=lambda pair: (pair[0], pair[1])) # Get the lexicographically largest pair

    # Merge tokens
    for word_id_this in byte_pair_index[max_frequency_pair]:
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
        word_id_set[word_id_this] = new_word_encoded_this
            
