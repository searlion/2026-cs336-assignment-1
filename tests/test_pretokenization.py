import pytest

from tokenizer.util import tokenizing_chunk

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