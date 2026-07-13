def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join(bytestring.decode("utf-8"))
print(decode_utf8_bytes_to_str_wrong("こ".encode("utf-8")))
