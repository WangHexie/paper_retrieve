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
from src.data.read_data import read_paper, read_validation_data, root_dir, read_train_data


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
    validation_description_tfidf = validation_to_to_tokenizer(read_validation_data()["description_text"].tolist(), tokenizer)
    train_description_tfidf = validation_to_to_tokenizer(read_train_data()["description_text"].tolist(), tokenizer)
    print("dasfsdfasdfsdfsdafasdfdsf")

    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"), paper_tfidf)
    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"), validation_description_tfidf)
    scipy.sparse.save_npz(os.path.join(root_dir(), "models", "train_description_tf_idf.npz"), train_description_tfidf)

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


def calculate_similarity(paper_name="paper_tf_idf.npz", description_name="description_tf_idf.npz", save_name="top_index.pk"):
    if not os.path.exists(os.path.join(root_dir(), "models", paper_name)) or not os.path.exists(
            os.path.join(root_dir(), "models", description_name)):
        output_tfidf_data()

    paper_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", paper_name))
    description_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", description_name))

    final = output_top_index(description_tf_idf, paper_tf_idf, 10, 500)
    with open(os.path.join(root_dir(), "models", save_name), "wb") as f:
        pickle.dump(final, f)


def save_prediction_of_xx(top_index_name="top_index.pk", is_validation=True):
    if not os.path.exists(os.path.join(root_dir(), "models", top_index_name)):
        calculate_similarity()
    with open(os.path.join(root_dir(), "models", top_index_name), "rb") as f:
        top_index = pickle.load(f)

    final_prediction = []

    paper_id = read_paper()["paper_id"]
    name_to_save = "validation.csv"

    if is_validation:
        description_id = read_validation_data()["description_id"].tolist()
    else:
        description_id = read_train_data()["description_id"].tolist()
        name_to_save = "train.csv"

    for i in range(len(top_index)):
        final_prediction.append([description_id[i]]+paper_id[top_index[i]].tolist()[:3])

    pd.DataFrame(final_prediction).to_csv(os.path.join(root_dir(), "result", name_to_save), index=False, header=False)


def prediction_metrics(true_labels, predictions):
    full_length = len(true_labels)
    correct_number = 0
    for i in range(full_length):
        if true_labels[i] in predictions[i]:
            correct_number+=1
    print(correct_number/full_length)
    print(correct_number, full_length)


def evaluate_on_train():
    calculate_similarity(description_name="train_description_tf_idf.npz", save_name="train_top_index.pk")
    save_prediction_of_xx(top_index_name="train_top_index.pk", is_validation=False)
    true_labels = read_train_data()["paper_id"].tolist()
    predictions = pd.read_csv(os.path.join(root_dir(), "result", "train.csv")).values.tolist()
    prediction_metrics(true_labels, predictions)


if __name__ == '__main__':
    # save_prediction_of_xx()
    evaluate_on_train()