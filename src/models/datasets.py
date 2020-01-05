import os

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchnlp import word_to_vector

from src.data.datasets import preprocessor, tokenize
from src.data.read_data import read_paper, read_train_data, root_dir, load_file_or_model


class TripletText(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, batch_size=64, sample_num=4, max_len=50, random=True, hard=200, use_idf=False):
        papers = read_paper()
        papers.dropna(subset=["paper_id"], inplace=True)
        papers["full"] = (papers["title"].apply(lambda x: x + " " if not pd.isna(x) else " ") + \
                          papers["journal"].apply(lambda x: x + " " if not pd.isna(x) else " ") + \
                          papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " ") + \
                          papers["abstract"].apply(lambda x: x + " " if not pd.isna(x) else " ")).apply(
            lambda x: preprocessor(x))
        papers.set_index("paper_id", inplace=True)
        papers.drop(columns=["keywords", "abstract", "journal", "title", "year"], inplace=True)

        self.papers = papers
        train_data = read_train_data()
        train_data.dropna(subset=["paper_id", "description_text"], inplace=True)
        self.train_pair = train_data["paper_id"]
        self.train_description = train_data["description_text"].apply(
            lambda x: preprocessor(x) if not pd.isna(x) else "  ")

        self.random = random
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.to_max = True
        self.embedding = word_to_vector.FastText(cache=os.path.join(root_dir(), "models"))
        self.max_len = max_len
        self.negative_sample = load_file_or_model("top_index_triplet.pk")
        self.hard = hard
        self.inverse_document_frequency = load_file_or_model("paper_inverse_frequency.pk")
        self.use_idf = use_idf

    def shuffle(self):
        self.train_description, self.train_pair, self.negative_sample = shuffle(self.train_description, self.train_pair,
                                                                                self.negative_sample)

    def max_index(self):
        return int(self.__len__() / self.batch_size)

    def string_to_vec(self, strings):
        wordss = list(map(lambda x: tokenize(x)[:self.max_len], strings))
        try:

            wordss = list(map(lambda x: x + ["nan"], wordss))
            if self.use_idf:
                embeddings = list(map(lambda words: torch.stack([self.embedding[word] * self.inverse_document_frequency[
                    word] if (word in self.inverse_document_frequency) else self.embedding[word] for word in words]),
                                      wordss))
            else:
                embeddings = list(map(lambda words: torch.stack([self.embedding[word] for word in words]), wordss))
        except RuntimeError:
            print(wordss)
            for i in wordss:
                if len(i) == 0:
                    print(i)

        fixed_length_embedding = pad_sequence(embeddings).permute(1, 2, 0)

        length_to_pad = self.max_len - fixed_length_embedding.shape[2]
        if length_to_pad > 0:
            shape = fixed_length_embedding.shape

            target = torch.zeros(shape[0], shape[1], self.max_len)
            target[:, :, :shape[2]] = fixed_length_embedding
            fixed_length_embedding = target

        return fixed_length_embedding

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > self.__len__():
            index = index % (self.max_index())
            if self.to_max:
                print("TO MAX")
                self.to_max = False

        anchors = self.train_description[index * self.batch_size:(index + 1) * self.batch_size].values
        positive_samples_index = self.train_pair[index * self.batch_size:(index + 1) * self.batch_size]
        positive_samples = self.papers.loc[positive_samples_index].values
        if self.random:
            temp = []
            for i in range(self.batch_size):
                negative_index = np.random.choice(self.papers.index, self.sample_num)
                while (negative_index == positive_samples_index.iloc[i]).any():
                    negative_index = np.random.choice(self.papers.index, self.sample_num)
                temp.append(self.papers.loc[negative_index].values)
            negative_samples = np.vstack(temp)
        else:
            # TODO: hard mining
            temp = []
            for i in range(self.batch_size):
                negative_index = np.random.choice(self.negative_sample[index * self.batch_size + i][:self.hard],
                                                  self.sample_num)
                negative_index = np.array(self.papers.index)[negative_index]
                while (negative_index == positive_samples_index.iloc[i]).any():
                    negative_index = np.random.choice(self.negative_sample[index * self.batch_size + i][:self.hard],
                                                      self.sample_num)

                    negative_index = np.array(self.papers.index)[negative_index]
                temp.append(self.papers.loc[negative_index].values)
            negative_samples = np.vstack(temp)
        return self.string_to_vec(anchors), self.string_to_vec(positive_samples.flatten()), self.string_to_vec(
            negative_samples.flatten())

    def __len__(self):
        return len(self.train_pair)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


if __name__ == '__main__':
    t = TripletText(random=False)
    t[0]
