import os
import re
import string
from typing import List

import numpy as np
import pandas as pd
from nltk import stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from src.data.read_data import root_dir, read_paper, read_train_data, read_validation_data

stemmer = stem.PorterStemmer()


def lower_case(data: str) -> str:
    return data.lower()


def remove_numbers(data: str) -> str:
    return re.sub(r'\d+', '', data)


def replace_punctuation(data: str) -> str:
    return data.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))


def remove_white_space(data: str) -> str:
    return data.strip()


def tokenize(data: str) -> List[str]:
    try:
        return word_tokenize(data)
    except LookupError:
        import nltk
        nltk.download('punkt')
        return word_tokenize(data)


def remove_stop_word(data: List[str]) -> List[str]:
    return [word for word in data if word not in ENGLISH_STOP_WORDS]


def stemming(data: List[str]) -> List[str]:
    # TODO: find better stemmer
    return [stemmer.stem(word) for word in data]


def text_preprocessing(data: str) -> List[str]:
    process_func = [lower_case, remove_numbers, replace_punctuation, remove_white_space,
                    tokenize,
                    remove_stop_word, stemming]

    for func in process_func:
        data = func(data)

    return data


def preprocessor(data: str) -> str:
    process_func = [lower_case, remove_numbers, replace_punctuation, remove_white_space]
    for func in process_func:
        data = func(data)
    return data


def tokenizer_and_after(data: str) -> List[str]:
    """
    for sklearn tfidf function use only
    :param data:
    :return:
    """
    process_func = [tokenize,
                    remove_stop_word]

    for func in process_func:
        data = func(data)

    return data


def list_to_str(x):
    return " ".join(x)


def make_dataset_for_fasttext():
    papers = read_paper()
    papers.dropna(subset=["paper_id"], inplace=True)
    a = (papers["title"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
         papers["journal"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
         papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " nan ") + \
         papers["abstract"].apply(lambda x: x + " " if not pd.isna(x) else " nan ")).apply(
        lambda x: preprocessor(x))

    b = read_train_data()
    b.dropna(subset=["description_text"], inplace=True)
    b = b["description_text"]

    c = read_validation_data()
    c.dropna(subset=["description_text"], inplace=True)
    c = c["description_text"]

    with open(os.path.join(root_dir(), "data", "paper.txt"), "w", encoding="utf-8") as f:
        for document in [a, b, c]:
            for index, item in document.items():
                if item == "NO_CONTENT" or item == np.nan:
                    continue
                f.write(item)
                f.write("\n")


if __name__ == '__main__':
    test_text = "i don't know that you are talking about.I knew you had a few apples which is 6 kg.\n"
    print(text_preprocessing(test_text))
    make_dataset_for_fasttext()
