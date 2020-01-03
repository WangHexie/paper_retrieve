import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.data.datasets import preprocessor
from src.data.read_data import root_dir, read_validation_data
from src.models.datasets import TripletText


def load_file_or_model(name):
    with open(os.path.join(root_dir(), "models", name), "rb") as f:
        model = pickle.load(f)
    return model


def save_embedding(vector: np.array(), name):
    with open(os.path.join(root_dir(), "models", name), "wb") as f:
        pickle.dump(vector, f)


def predict(model, data):
    """
    load model first
    :param model:
    :return:
    """
    device = torch.device('cuda')
    model.to(device)
    data.to(device)
    return model.transform(data)


def output_description_matrix(description, model_name, triplet_text: TripletText):
    sub_length = 256
    length = len(description)

    dt = triplet_text
    model = load_file_or_model(model_name)

    full_embedding = []
    start = 0
    while length > 0:
        if length < sub_length:
            sub_length = length
        length -= sub_length
        vector = dt.string_to_vec(description[start: start + sub_length])
        full_embedding.append(model(vector).numpy())

    return np.stack(full_embedding)


def output_description_and_paper_text(model_name):
    triplet_text = TripletText()
    paper_info = output_description_matrix(triplet_text.papers["full"].values, model_name, triplet_text)
    save_embedding(paper_info, "paper_info_triplet.pk")
    save_embedding(np.array(triplet_text.papers.index), "paper_id.pk")

    print("one finished")

    train_description = output_description_matrix(triplet_text.train_description["description_text"].values, model_name,
                                                  triplet_text)
    save_embedding(train_description, "train_description.pk")
    save_embedding(triplet_text.train_pair["paper_id"].values, "train_paper_id.pk")

    print("one finished")

    validation_data = read_validation_data()["description_text"].apply(
        lambda x: preprocessor(x) if not pd.isna(x) else "  ").values

    validation_vector = output_description_matrix(validation_data, model_name, triplet_text)
    save_embedding(validation_vector, "validation_description.pk")
    save_embedding(validation_data["description_id"], "validation_id.pk")

    print("one finished")


output_top_index


