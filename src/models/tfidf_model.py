import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.datasets import text_preprocessing, preprocessor, list_to_str
from src.data.read_data import read_paper, read_validation_data, root_dir


def tf_idf_data(papers: pd.DataFrame, fast=True):
    vectorizer = TfidfVectorizer(max_features=2000, preprocessor=preprocessor)
    if fast:
        print("faster")
        strings = papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " nan ") + \
                  papers["abstract"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
                  papers["journal"].apply(lambda x: x + " " if not pd.isna(x) else " nan ")
    else:
        strings = (papers["keywords"].apply(lambda x: x.replace(";", " ") if not pd.isna(x) else str(x)) + papers[
            "abstract"].apply(lambda x: str(x)) + papers["journal"]).apply(
            lambda x: list_to_str(text_preprocessing(x)) if not pd.isna(x) else str(x))
    print("to lsit")
    strings = strings.tolist()
    print("start vectorize")
    matrix = vectorizer.fit_transform(strings)
    return matrix, vectorizer


def validation_to_to_tokenizer(papers: List[str], vectorizer: TfidfVectorizer):
    return vectorizer.transform(papers)


def matrix_similarity(description_matrix, paper_matrix):
    return cosine_similarity(description_matrix, paper_matrix, dense_output=False)


def output_tfidf_data():
    paper_tfidf, tokenizer = tf_idf_data(read_paper())
    description_tfidf = validation_to_to_tokenizer(read_validation_data()["description_text"].tolist(), tokenizer)
    print("dasfsdfasdfsdfsdafasdfdsf")

    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"), paper_tfidf)
    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"), description_tfidf)
    with open(os.path.join(root_dir(), "models", "tokenizer.pk"), "wb") as f:
        pickle.dump(tokenizer, f)


def output_top_index(description_tf_idf, paper_tf_idf, top=3, sub_length=1000):
    length = description_tf_idf.shape[0]

    start = 0
    full_max_columns = []
    print("start")
    while length > 0:
        if length < sub_length:
            sub_length = length
        length -= sub_length

        result = matrix_similarity(description_tf_idf[start:start + sub_length], paper_tf_idf)

        max_cols = []

        for _ in range(top):
            col = result.argmax(1)
            col = col.flatten().tolist()[0]
            value = result.max(1)
            value = value.todense().flatten().tolist()[0]
            row = list(range(sub_length))

            max_matrix = scipy.sparse.csr_matrix((value, (row, col)), result.shape)
            result -= max_matrix

            max_cols.append(col)

        max_cols = np.array(max_cols).T.tolist()
        full_max_columns += max_cols

        print("length left:", length)
        start += sub_length

    return full_max_columns


def calculate_similarity():
    """
    UNFINISHED
    :return:
    """
    paper_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"))
    description_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"))

    final = output_top_index(description_tf_idf, paper_tf_idf, 10, 500)
    with open(os.path.join(root_dir(), "models", "top_index"), "wb") as f:
        pickle.dump(final, f)


if __name__ == '__main__':
    calculate_similarity()
