import io
import os

from src.data.read_data import root_dir


def load_vectors(fname, number_to_read=100000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    number = 0
    for line in fin:
        number += 1
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        if number == number_to_read:
            break
    return data


if __name__ == '__main__':
    print(root_dir())
    a = load_vectors(os.path.join(root_dir(), "models", "wiki-news-300d-1M.vec"))
    print(a)
    print(a["cornmeal"])
