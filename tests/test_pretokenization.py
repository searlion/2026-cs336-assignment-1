import pytest

from tokenizer.util import *

@pytest.mark.parametrize("original_string, expected_tokens", [
    ("some text that i'll pre-tokenize", ["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]),
])
def test_tokenizing_chunk(original_string, expected_tokens):
    tokens = [match.group() for match in tokenizing_chunk(original_string)]
    assert tokens == expected_tokens


@pytest.mark.parametrize("chunk", [
    "",
    "   ",
    "  hello  world  ",
    "don't 123 !!!",
    "café 你好 🎉",
])
def test_pretokenizing_round_trip(chunk):
    tokens = [match.group() for match in tokenizing_chunk(chunk)]
    # Reconstruct the original chunk from the tokens
    reconstructed_chunk = "".join(tokens)
    assert reconstructed_chunk == chunk

@pytest.mark.parametrize("input_string, expected_output", [
    ("", b""),
    ("   ", b"   "),
    ("  hello  world  ", b"  hello  world  "),
    ("don't 123 !!!", b"don't 123 !!!"),
    ("café", b"caf\xc3\xa9"),
])
def test_encode_as_hashable_bytes(input_string, expected_output):
    assert encode_as_hashable_bytes(input_string) == expected_output

@pytest.mark.parametrize("input_string, expected_output", [
    ("hello", [b"h", b"e", b"l", b"l", b"o"]),
    ("café", [b"c", b"a", b"f", b"\xc3", b"\xa9"]),
])
def test_encode_as_bytes(input_string, expected_output):
    assert encode_as_bytes(input_string.encode("utf-8")) == expected_output

@pytest.mark.parametrize("encoded_word, expected_counter", [
    ([b"h", b"e", b"l", b"l", b"o"], {(b"h", b"e"): 1, (b"e", b"l"): 1, (b"l", b"l"): 1, (b"l", b"o"): 1}),
    ([b"a", b"b", b"a", b"b"], {(b"a", b"b"): 2, (b"b", b"a"): 1}),
    ([b"a", b"a", b"a"], {(b"a", b"a"): 2}),
])
def test_pairs_from_encoded_word(encoded_word, expected_counter):
    assert pairs_from_encoded_word(encoded_word) == expected_counter

@pytest.mark.parametrize("straight_index, expected_max_frequency_pair, expected_max_frequency", [
    ({(b"a", b"b"): 2, (b"b", b"a"): 1}, (b"a", b"b"), 2),
    ({(b"a", b"b"): 1, (b"z", b"a"): 2}, (b"z", b"a"), 2),
])
def test_calculate_lexicographically_largest_byte_pair_with_highest_frequency(straight_index, expected_max_frequency_pair, expected_max_frequency):
    max_frequency_pair, max_frequency = calculate_lexicographically_largest_byte_pair_with_highest_frequency(straight_index)
    assert (max_frequency_pair, max_frequency) == (expected_max_frequency_pair, expected_max_frequency)