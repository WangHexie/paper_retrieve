import os
from typing import List

import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.datasets import text_preprocessing, preprocessor, tokenizer_and_after
from src.data.read_data import read_paper, read_validation_data, root_dir
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def list_to_str(x):
    return " ".join(x)


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


def matrix_multiplication(description_matrix, paper_matrix):
    return cosine_similarity(description_matrix, paper_matrix, dense_output=False)


def output_tfidf_data():
    paper_tfidf, tokenizer = tf_idf_data(read_paper())
    description_tfidf = validation_to_to_tokenizer(read_validation_data()["description_text"].tolist(), tokenizer)
    print("dasfsdfasdfsdfsdafasdfdsf")

    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"), paper_tfidf)
    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"), description_tfidf)


def calculate_similarity():


if __name__ == '__main__':
    a, b = tf_idf_data(read_paper())
    p = validation_to_to_tokenizer(read_validation_data()["description_text"].tolist(), b)
    print("dasfsdfasdfsdfsdafasdfdsf")

    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"), a)
    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"), p)

    print("dasfaaaaaaaaaa")
    xxxx = matrix_multiplication(p, a)
    print(xxxx)


