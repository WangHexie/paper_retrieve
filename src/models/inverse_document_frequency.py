from typing import List

import math
import pandas as pd

from src.data.datasets import preprocessor, tokenize
from src.data.read_data import read_paper, save_embedding


def calculate_inverse_document_frequency(document: List[str]):
    full = dict()
    number_of_document = len(document)

    print(number_of_document)

    result = map(lambda x: preprocessor(x), document)
    result = map(lambda x: tokenize(x), result)

    for string in result:
        counted = set()
        for word in string:
            if word not in counted:
                try:
                    full[word] += 1
                except KeyError as e:
                    full[word] = 1
                    counted.add(word)

    print(full["is"])

    for key in full.keys():
        full[key] = math.log2(number_of_document / full[key])

    print(full["is"])

    return full


def output_idf_of_paper():
    papers = read_paper()
    full_string = (papers["title"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
                   papers["journal"].apply(lambda x: x + " " if not pd.isna(x) else " nan ") + \
                   papers["keywords"].apply(lambda x: x.replace(";", " ") + " " if not pd.isna(x) else " nan ") + \
                   papers["abstract"].apply(lambda x: x + " " if not pd.isna(x) else " nan "))
    final = calculate_inverse_document_frequency(full_string.values)
    save_embedding(final, "paper_inverse_frequency")


if __name__ == '__main__':
    output_idf_of_paper()
