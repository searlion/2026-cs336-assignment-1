from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        for result in p.imap_unordered(f, [1, 2, 3]):
            print(result)