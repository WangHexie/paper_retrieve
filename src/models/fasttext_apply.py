import os

import fasttext

from src.data.read_data import root_dir

def train():
    model = fasttext.train_unsupervised(os.path.join(root_dir(), "data", "paper.txt"), model='skipgram')
    model.save_model(os.path.join(root_dir(), "models", "fasttext"))

if __name__ == '__main__':
    model = fasttext.load_model(os.path.join(root_dir(), "models", "fasttext.bin"))
    print(model.get_word_vector("the"))
    print(model.get_word_vector("non"))
    print(model.get_word_vector("none"))
    print(model.get_nearest_neighbors("is"))
    print(model.get_word_vector("porkkk"))
    print(model.get_nearest_neighbors("science"))


