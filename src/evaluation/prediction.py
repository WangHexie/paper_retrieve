import os

import numpy as np
import pandas as pd
import torch

from src.config.configs import triplet_config, default_train_config
from src.data.datasets import preprocessor
from src.data.read_data import read_validation_data, load_file_or_model, save_embedding, root_dir, read_train_data
from src.evaluation.evalutation_func import output_top_index, save_prediction_of_xx_triplet, prediction_metrics
from src.models.datasets import TripletText

config = triplet_config


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
        print("length left", length, length / org)
        if length < sub_length:
            sub_length = length
        length -= sub_length
        vector = dt.string_to_vec(description[start: start + sub_length])
        full_embedding.append(model.transform(vector).cpu().detach().numpy())

        start += sub_length

    return np.concatenate(full_embedding)


def output_description_and_paper_text(model_name, temporary=False, number=1000):
    triplet_text = TripletText(default_train_config.batch_size,
                               default_train_config.sample_number,
                               random=default_train_config.random,
                               hard=default_train_config.hard,
                               max_len=default_train_config.max_len,
                               use_idf=default_train_config.use_idf,
                               use_self_train=default_train_config.use_self_train)

    paper_info = output_description_matrix(triplet_text.papers["full"].values, model_name, triplet_text)
    save_embedding(paper_info, config.paper_embedding)
    save_embedding(np.array(triplet_text.papers.index), config.paper_id)

    print("one finished")

    train_description = output_description_matrix(triplet_text.train_description.values, model_name,
                                                  triplet_text)
    save_embedding(train_description, config.train_description_embedding)
    save_embedding(triplet_text.train_pair.values, config.train_paper_id)
    train_data = read_train_data()
    train_data.dropna(subset=["paper_id", "description_text"], inplace=True)
    save_embedding(train_data["description_id"].values, config.train_description_id)

    print("one finished")
    validation_data = read_validation_data()
    validation_text = validation_data["description_text"].apply(
        lambda x: preprocessor(x) if not pd.isna(x) else " nan ").values

    # wordss = list(map(lambda x: tokenize(x), validation_text))
    # index_to_drop = list(filter(lambda x: x > -1, [i if len(wordss[i]) == 0 else -1 for i in range(len(wordss))]))
    # # print(index_to_drop)
    # if len(index_to_drop) > 0:
    #     validation_data["description_id"] = validation_data["description_id"].drop(index_to_drop)

    validation_vector = output_description_matrix(validation_text, model_name, triplet_text)
    save_embedding(validation_vector, config.validation_description_embedding)
    save_embedding(validation_data["description_id"], config.validation_description_id)

    print("one finished")


def prediction_nn(model_name, top=100):
    # output_description_and_paper_text(model_name)
    full_max = output_top_index(load_file_or_model(config.train_description_embedding),
                                load_file_or_model(config.paper_embedding),
                                top=top, dense=True, sub_length=1000, fast=True)
    save_embedding(full_max, config.train_top_index)
    save_prediction_of_xx_triplet(config.train_top_index, is_validation=False, name_to_save=config.train_prediction,
                                  number_to_save=top)
    df = pd.read_csv(os.path.join(root_dir(), "result", config.train_prediction), header=None)
    prediction_metrics(load_file_or_model(config.train_paper_id), df.values)

    full_max = output_top_index(load_file_or_model(config.validation_description_embedding),
                                load_file_or_model(config.paper_embedding), top=3, dense=True)
    save_embedding(full_max, config.validation_top_index)
    save_prediction_of_xx_triplet(config.validation_top_index, is_validation=True,
                                  name_to_save=config.validation_prediction)


def output_fast_vector():
    triplet_text = TripletText(0, 0, random=True, hard=-1, max_len=100, use_idf=True, use_self_train=True)

    paper_info = triplet_text.string_to_1d_vec(triplet_text.papers["full"].values)
    save_embedding(paper_info, config.paper_embedding)
    save_embedding(np.array(triplet_text.papers.index), config.train_paper_id)

    print("one finished")

    train_description = triplet_text.string_to_1d_vec(triplet_text.train_description.values)
    save_embedding(train_description, config.train_description_embedding)
    save_embedding(triplet_text.train_pair.values, config.train_paper_id)
    train_data = read_train_data()
    train_data.dropna(subset=["paper_id", "description_text"], inplace=True)
    save_embedding(train_data["description_id"].values, config.train_description_id)

    print("one finished")
    validation_data = read_validation_data()
    validation_text = validation_data["description_text"].apply(
        lambda x: preprocessor(x) if not pd.isna(x) else " nan ").values

    # wordss = list(map(lambda x: tokenize(x), validation_text))
    # index_to_drop = list(filter(lambda x: x > -1, [i if len(wordss[i]) == 0 else -1 for i in range(len(wordss))]))
    # # print(index_to_drop)
    # if len(index_to_drop) > 0:
    #     validation_data["description_id"] = validation_data["description_id"].drop(index_to_drop)

    validation_vector = triplet_text.string_to_1d_vec(validation_text)
    save_embedding(validation_vector, config.validation_description_embedding)
    save_embedding(validation_data["description_id"], config.validation_description_id)

    print("one finished")


def fasttext_prediction():
    output_fast_vector()
    full_max = output_top_index(load_file_or_model(config.train_description_embedding),
                                load_file_or_model(config.paper_embedding),
                                top=3000, dense=True, sub_length=1000)
    save_embedding(full_max, config.train_top_index)
    save_prediction_of_xx_triplet(config.train_top_index,
                                  is_validation=False,
                                  name_to_save=config.train_prediction,
                                  number_to_save=10)
    df = pd.read_csv(os.path.join(root_dir(), "result", config.train_prediction), header=None)
    prediction_metrics(load_file_or_model(config.train_paper_id), df.values)

    full_max = output_top_index(load_file_or_model(config.validation_description_embedding),
                                load_file_or_model(config.paper_embedding), top=3, dense=True)
    save_embedding(full_max, config.validation_top_index)
    save_prediction_of_xx_triplet(config.validation_top_index, is_validation=True,
                                  name_to_save=config.validation_prediction)

if __name__ == '__main__':
    prediction_nn("modelhardest2_abs_loss_idf4.pk", 3)
