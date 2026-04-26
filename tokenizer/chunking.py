from cs336_basics import pretokenization_example
from multiprocessing import Pool
from os import cpu_count

def read_chunk(tuple):
    with open(tuple[0], "rb") as f:
        f.seek(tuple[1])
        chunk = f.read(tuple[2] - tuple[1]).decode("utf-8", errors="ignore")
        return chunk

if __name__ == "__main__":
    num_processes = cpu_count()
    with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    with Pool(num_processes) as p:
        for chunk in p.imap_unordered(read_chunk, [("../data/TinyStoriesV2-GPT4-valid.txt", start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]):
            print(chunk)

    # with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    #     num_processes = cpu_count()
    #     boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    #     # The following is a serial implementation, but you can parallelize this
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #         print(start, end)
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token