import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


type word_id = int 
type frequency = int
type byte_pair = tuple[bytes, bytes]
type encoded_word = list[bytes] # A list of lists of bytes, where each list of bytes is a pre-tokenized word
type encoded_word_hashable = bytes # A list of bytes, where each byte is a pre-tokenized word

# Pre-tokenization
def tokenizing_chunk(chunk):
    return re.finditer(PAT, chunk)

# # Encode the pre-tokenized words as bytes
def encode_as_hashable_bytes(chunk):
    return chunk.encode("utf-8")

def encode_as_bytes(bytestring: bytes):
    return [bytes([b]) for b in bytestring]

if __name__ == "__main__":
    chunk = "some text that i'll pre-tokenize"
    ## initialization of data structures for counting byte pairs and word pairs
    byte_pair_index : dict[byte_pair, set[word_id]] = {} # an inverted index telling you where each pair lives, so after a merge you only touch affected words
    straight_index : dict[byte_pair, frequency] = {} # The thing you argmax over to pick the next merge.
    word_id_set : dict[word_id, encoded_word] = {} # Each unique pre-token gets an integer id.
    word_id_set_reverse : dict[encoded_word_hashable, word_id] = {} # Reverse lookup for the above
    word_id_count : dict[word_id, frequency] = {} # How many times that word occured (its frequency weight)

    ## Below code assigns a word_id to each pre-tokenized word, and counts its frequency
    for match in tokenizing_chunk(chunk):
        encoded_hashable = encode_as_hashable_bytes(match.group())
        encoded = encode_as_bytes(encoded_hashable)
        if encoded_hashable not in word_id_set_reverse.keys():
            word_id_this = len(word_id_set)
            word_id_set[word_id_this] = encoded
            word_id_set_reverse[encoded_hashable] = word_id_this
            word_id_count[word_id_this] = 1
        else:
            word_id_this = word_id_set_reverse[encoded_hashable]
            word_id_count[word_id_this] += 1
    
    ## Below code builds the byte pair index, straight index, and reverse index for the pre-tokenized words
    for word_id_this in word_id_set.keys():
        word_id_this_frequency = word_id_count[word_id_this]
        word_encoded_this = word_id_set[word_id_this]
        for tuple_pair in zip(word_encoded_this[:-1], word_encoded_this[1:]):
            straight_index[tuple_pair] = straight_index.get(tuple_pair, 0) + word_id_this_frequency
            byte_pair_index.setdefault(tuple_pair, set()).add(word_id_this)
        

    ## Get the lexicographically largest byte pair with the highest frequency
    max_frequency_pair, max_frequency = max(straight_index.items(), key=lambda item: (item[1], item[0][0], item[0][1])) # Get the lexicographically largest pair with the highest frequency
    print(max_frequency_pair, max_frequency)

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
            
