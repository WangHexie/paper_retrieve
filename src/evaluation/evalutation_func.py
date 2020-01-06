import os
import pickle

import pandas as pd
import numpy as np
import scipy
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from torch import nn
import torch
import torch.nn.functional as F

from src.config.configs import triplet_config
from src.data.read_data import root_dir, read_paper, read_validation_data, read_train_data, load_file_or_model

def cs(x,y):
    prod = torch.mm(x, y)
    len1 = torch.sqrt(torch.mm(x, x.transpose(1,0)))
    len2 = torch.sqrt(torch.mm(y, y.transpose(1,0)))
    return prod / (len1 * len2)


def sort_func(matrix, top):
    device = torch.device('cuda')
    tm = torch.FloatTensor(matrix)
    tm.to(device)
    for _ in range(top):
        col = tm.argmax(1)
        col = col.cpu().flatten().tolist()
        value = tm.max(1)

        value = value.flatten().tolist()
        row = list(range(sub_length))

        # max_matrix = scipy.sparse.csr_matrix((value, (row, col)), result.shape)
        max_matrix = np.zeros(result.shape)
        max_matrix[row, col] = value
        result -= max_matrix
        max_cols.append(col)


def output_top_index(description_tf_idf, paper_tf_idf, top=3, sub_length=10, dense=False):
    length = description_tf_idf.shape[0]

    start = 0
    full_max_columns = []
    print("start")
    # if dense:
    #     device = torch.device('cuda')
    #     cos = nn.CosineSimilarity(dim=1)
    #     cos.to(device)
    #     description_tf_idf = torch.FloatTensor(description_tf_idf)
    #     paper_tf_idf = torch.FloatTensor(paper_tf_idf)
    #     paper_tf_idf.to(device)
    #     description_tf_idf.to(device)

    while length > 0:
        if length < sub_length:
            sub_length = length
        length -= sub_length

        if not dense:
            result = matrix_similarity(description_tf_idf[start:start + sub_length], paper_tf_idf, dense)
        else:
            # result = F.cs(description_tf_idf[start:start + sub_length], paper_tf_idf)
            result = matrix_similarity(description_tf_idf[start:start + sub_length], paper_tf_idf, dense)

        max_cols = []

        if not dense:
            for _ in range(top):
                col = result.argmax(1)
                col = col.flatten().tolist()[0]
                value = result.max(1)
                try:
                    value = value.todense().flatten().tolist()[0]
                except AttributeError:
                    value = value.flatten().tolist()[0]
                row = list(range(sub_length))

                max_matrix = scipy.sparse.csr_matrix((value, (row, col)), result.shape)
                result -= max_matrix
                max_cols.append(col)

        else:
            # for _ in range(top):
            #     col = result.argmax(1)
            #     col = col.flatten().tolist()
            #     value = result.max(1)
            #
            #     value = value.flatten().tolist()
            #     row = list(range(sub_length))
            #
            #     # max_matrix = scipy.sparse.csr_matrix((value, (row, col)), result.shape)
            #     max_matrix = np.zeros(result.shape)
            #     max_matrix[row, col] = value
            #     result -= max_matrix
            #     max_cols.append(col)
            print("start sort")
            col = result.argsort(1)[:, :top]
            col = col.tolist()

            max_cols = col

        # max_cols = np.array(max_cols).T.tolist()

        full_max_columns += max_cols

        print("length left:", length)
        start += sub_length

    return full_max_columns


def calculate_similarity(paper_name="paper_tf_idf.npz", description_name="description_tf_idf.npz",
                         save_name="top_index.pk", sub_length=500):
    # if not os.path.exists(os.path.join(root_dir(), "models", paper_name)) or not os.path.exists(
    #         os.path.join(root_dir(), "models", description_name)):
    #     output_tfidf_data()

    paper_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", paper_name))
    description_tf_idf = scipy.sparse.load_npz(os.path.join(root_dir(), "models", description_name))

    final = output_top_index(description_tf_idf, paper_tf_idf, 10, sub_length)
    with open(os.path.join(root_dir(), "models", save_name), "wb") as f:
        pickle.dump(final, f)


def save_prediction_of_xx(top_index_name="top_index.pk", is_validation=True, number_to_save=3, name_to_save="validation.csv"):
    if not os.path.exists(os.path.join(root_dir(), "models", top_index_name)):
        calculate_similarity()
    with open(os.path.join(root_dir(), "models", top_index_name), "rb") as f:
        top_index = pickle.load(f)

    final_prediction = []

    paper_id = read_paper()["paper_id"]

    if is_validation:
        description_id = read_validation_data()["description_id"].tolist()
    else:
        description_id = read_train_data()["description_id"].tolist()

    for i in range(len(top_index)):
        final_prediction.append([description_id[i]] + paper_id[top_index[i]].tolist()[:number_to_save])

    pd.DataFrame(final_prediction).to_csv(os.path.join(root_dir(), "result", name_to_save), index=False, header=False)


def prediction_metrics(true_labels, predictions):
    full_length = len(true_labels)
    correct_number = 0
    for i in range(full_length):
        if true_labels[i] in predictions[i]:
            correct_number += 1
    print(correct_number / full_length)
    print(correct_number, full_length)


def matrix_similarity(description_matrix, paper_matrix, dense):

    return pairwise_distances(description_matrix, paper_matrix,metric="l1", n_jobs=4)
    # return cosine_similarity(description_matrix, paper_matrix, dense_output=dense)


def save_prediction_of_xx_triplet(top_index_name="top_index.pk", is_validation=True, number_to_save=3, name_to_save="validation.csv"):
    top_index = load_file_or_model(top_index_name)

    final_prediction = []
    paper_id = load_file_or_model("paper_id.pk")

    if is_validation:
        description_id = load_file_or_model("validation_id.pk")
    else:
        description_id = load_file_or_model("train_description_id.pk")

    for i in range(len(top_index)):
        final_prediction.append([description_id[i]] + paper_id[top_index[i]].tolist()[:number_to_save])

    pd.DataFrame(final_prediction).to_csv(os.path.join(root_dir(), "result", name_to_save), index=False, header=False)