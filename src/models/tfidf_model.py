import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.configs import train_evaluate_shorter, train_evaluate
from src.data.datasets import text_preprocessing, preprocessor, list_to_str, tokenizer_and_after
from src.data.read_data import read_paper, read_validation_data, root_dir, read_train_data
from src.evaluation.evalutation_func import calculate_similarity, save_prediction_of_xx, prediction_metrics


def tf_idf_data(papers: pd.DataFrame, handle_type=0):
    """

    :param papers:
    :param handle_type: 0: key abs journal 1: all + stemming 2: keyword stemming
    :return:
    """
    if handle_type == 0:
        vectorizer = TfidfVectorizer(max_features=2000, preprocessor=preprocessor)

        print("faster")
        strings = papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " nan ") + \
                  papers["abstract"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
                  papers["journal"].apply(lambda x: x + " " if not pd.isna(x) else " nan ")
    elif handle_type == 1:
        vectorizer = TfidfVectorizer(max_features=2000, preprocessor=preprocessor)

        strings = (papers["keywords"].apply(lambda x: x.replace(";", " ") if not pd.isna(x) else str(x)) + papers[
            "abstract"].apply(lambda x: str(x)) + papers["journal"]).apply(
            lambda x: list_to_str(text_preprocessing(x)) if not pd.isna(x) else str(x))
    elif handle_type == 2:
        vectorizer = TfidfVectorizer(max_features=2000, preprocessor=preprocessor, tokenizer=tokenizer_and_after)

        strings = papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " ") + \
                  papers["title"].apply(lambda x: x + " " if not pd.isna(x) else " ")
    else:
        print("error")

    print("to list")
    strings = strings.tolist()
    print("start vectorize")
    matrix = vectorizer.fit_transform(strings)
    return matrix, vectorizer


def validation_to_to_tokenizer(papers: List[str], vectorizer: TfidfVectorizer):
    return vectorizer.transform(papers)


def output_tfidf_data(handle_type=0, config=None):
    paper_tfidf, tokenizer = tf_idf_data(read_paper(), handle_type=handle_type)
    validation_description_tfidf = validation_to_to_tokenizer(read_validation_data()["description_text"].tolist(),
                                                              tokenizer)
    train_description_tfidf = validation_to_to_tokenizer(
        read_train_data()["description_text"].apply(lambda x: str(x)).tolist(), tokenizer)
    print("dasfsdfasdfsdfsdafasdfdsf")

    tokenizer_name = "tokenizer.pk"
    if config is None:
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", "paper_tf_idf.npz"), paper_tfidf)
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", "description_tf_idf.npz"),
                              validation_description_tfidf)
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", "train_description_tf_idf.npz"),
                              train_description_tfidf)
    else:
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", config["paper_tf_idf_name"]), paper_tfidf)
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", config["description_tf_idf_name"]),
                              validation_description_tfidf)
        scipy.sparse.save_npz(os.path.join(root_dir(), "models", config["description_tf_idf_name"]),
                              train_description_tfidf)
        tokenizer_name = config["tokenizer_name"]

    with open(os.path.join(root_dir(), "models", tokenizer_name), "wb") as f:
        pickle.dump(tokenizer, f)


def evaluate_on_train(config):
    output_tfidf_data(handle_type=config["handle_type"], config=config)
    calculate_similarity(description_name=config["description_tf_idf_name"], save_name=config["top_index_save_name"],
                         sub_length=200)
    save_prediction_of_xx(top_index_name=config["top_index_save_name"], is_validation=False, name_to_save=config["result_name"])
    true_labels = read_train_data()["paper_id"].tolist()
    predictions = pd.read_csv(os.path.join(root_dir(), "result", config["result_name"]), header=None).values.tolist()
    prediction_metrics(true_labels, predictions)

if __name__ == '__main__':
    # save_prediction_of_xx()
    # evaluate_on_train(train_evaluate_shorter)
    config = train_evaluate_shorter
    # save_prediction_of_xx(top_index_name=config["top_index_save_name"], is_validation=False, number_to_save=15)
    true_labels = read_train_data()["paper_id"].tolist()
    predictions = pd.read_csv(os.path.join(root_dir(), "result", config["result_name"]), header=None).values.tolist()
    prediction_metrics(true_labels, predictions)

    # TODO try shorter using key and title and stemming 2.
