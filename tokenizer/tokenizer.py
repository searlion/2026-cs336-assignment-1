type Vocabulary = dict[int, bytes]
type Merges = list[tuple[bytes, bytes]]
type SpecialTokens = list[str] | None
type encoded_word = list[bytes] # A list bytes, where each bytes is a pre-tokenized word

from collections.abc import Iterable, Iterator
import json
import regex as re
from collections import Counter
from multiprocessing import Pool
from os import cpu_count
from functools import partial
from typing import BinaryIO
import os
from .util import encode_as_bytes, PAT
from functools import lru_cache

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def apply_merge(encoded_word: encoded_word, merge: tuple[bytes, bytes]) -> encoded_word:
    """
    Applies a single merge to an encoded word. If the merge is not applicable, returns the original encoded word.
    """
    merge_token_1, merge_token_2 = merge
    merged_token = merge_token_1 + merge_token_2
    new_encoded_word = []
    i = 0
    while i < len(encoded_word):
        if i < len(encoded_word) - 1 and encoded_word[i] == merge_token_1 and encoded_word[i + 1] == merge_token_2:
            new_encoded_word.append(merged_token)
            i += 2  # Skip the next token since it was merged
        else:
            new_encoded_word.append(encoded_word[i])
            i += 1
    return new_encoded_word

class Tokenizer:

    def __init__(self, vocab: Vocabulary, merges: Merges, special_tokens: SpecialTokens = None):
        original_vocab_size = len(vocab)
        vocab_inverse = {v: k for k, v in vocab.items()}
        special_tokens_filtered = [token for token in (special_tokens or []) if token.encode("utf-8") not in vocab_inverse]
        for i in range(original_vocab_size, original_vocab_size + len(special_tokens_filtered or [])):
            vocab[i] = special_tokens_filtered[i - original_vocab_size].encode("utf-8")
        self.vocab = vocab
        self.vocab_inverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }
        
        with open(merges_filepath, encoding="utf-8") as f:
            gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_reference_merges
            ]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:

        token_ids_list: list[int] = []
        # Step 1: split off special tokens first
        if not self.special_tokens:
            segments = [text]
        else:
            special_tokens_sorted = sorted(self.special_tokens or [], key=len, reverse=True)
            special_tokens_pattern ="(" + "|".join([re.escape(token) for token in special_tokens_sorted]) + ")"
            segments = re.split(special_tokens_pattern, text)

        # Step 2: Pre-tokenize
        for segment in segments:
            if segment in (self.special_tokens or []):
                # If the segment is a special token, encode it directly
                token_ids_list.append(self.vocab_inverse[segment.encode("utf-8")])
            else:
                # Otherwise, pre-tokenize the segment using the PAT regex pattern
                pretokens = re.finditer(PAT, segment)
                for pretoken in pretokens:
                    # Step 3a: encode the pre-token as bytes
                    encoded_pretoken = encode_as_bytes(pretoken.group().encode("utf-8"))
                    # Step 3b: apply merges to the encoded pre-token
                    for merge in self.merges:
                        encoded_pretoken = apply_merge(encoded_pretoken, merge)
                    # Step 3c: convert the encoded pre-token to a list of integer ids
                    token_ids = [self.vocab_inverse[token] for token in encoded_pretoken]
                    token_ids_list.extend(token_ids)

        return token_ids_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Implement the encoding logic for an iterable of strings here
        pass

    def decode(self, ids: list[int]) -> str:
        # Implement the decoding logic here
        output = b""
        for id in ids:
            output += self.vocab.get(id, b"")
        # Note: Using errors='replace' will automatically replace malformed data with U+FFFD marker
        return output.decode("utf-8", errors="replace")