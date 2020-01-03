import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.data.datasets import preprocessor
from src.data.read_data import root_dir, read_validation_data, read_train_data, load_file_or_model, save_embedding
from src.evaluation.evalutation_func import output_top_index, save_prediction_of_xx_triplet, prediction_metrics
from src.models.datasets import TripletText


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
    sub_length = 1024
    length = len(description)
    org = length

    dt = triplet_text
    model = load_file_or_model(model_name)

    full_embedding = []
    start = 0
    while length > 0:
        print("length left", length, length/org)
        if length < sub_length:
            sub_length = length
        length -= sub_length
        vector = dt.string_to_vec(description[start: start + sub_length])
        full_embedding.append(model.transform(vector).cpu().detach().numpy())

    return np.concatenate(full_embedding)


def output_description_and_paper_text(model_name):
    triplet_text = TripletText()
    # paper_info = output_description_matrix(triplet_text.papers["full"].values, model_name, triplet_text)
    # save_embedding(paper_info, "paper_info_triplet.pk")
    # save_embedding(np.array(triplet_text.papers.index), "paper_id.pk")
    #
    # print("one finished")

    train_description = output_description_matrix(triplet_text.train_description.values, model_name,
                                                  triplet_text)
    save_embedding(train_description, "train_description.pk")
    save_embedding(triplet_text.train_pair["paper_id"].values, "train_paper_id.pk")
    train_data = read_train_data()
    train_data.dropna(subset=["paper_id", "description_text"], inplace=True)
    save_embedding(train_data["description_id"].values, "train_description_id.pk")


    print("one finished")

    validation_data = read_validation_data()["description_text"].apply(
        lambda x: preprocessor(x) if not pd.isna(x) else "  ").values

    validation_vector = output_description_matrix(validation_data, model_name, triplet_text)
    save_embedding(validation_vector, "validation_description.pk")
    save_embedding(validation_data["description_id"], "validation_id.pk")

    print("one finished")


if __name__ == '__main__':
    output_description_and_paper_text("model8.pk")
    full_max = output_top_index(load_file_or_model("train_description.pk"), load_file_or_model("paper_info_triplet.pk"), top=3)
    save_embedding(full_max, "top_index_triplet.pk")
    save_prediction_of_xx_triplet("top_index_triplet.pk", is_validation=False, name_to_save="validation_triplet.csv")
    prediction_metrics(load_file_or_model("train_paper_id.pk"), pd.read_csv("validation_triplet.csv"))

    full_max = output_top_index(load_file_or_model("validation_description.pk"), load_file_or_model("paper_info_triplet.pk"), top=3)
    save_embedding(full_max, "top_index_triplet.pk")
    save_prediction_of_xx_triplet("top_index_triplet.pk", is_validation=False, name_to_save="validation_triplet.csv")
    prediction_metrics(load_file_or_model("train_paper_id.pk"), pd.read_csv("validation_triplet.csv"))



