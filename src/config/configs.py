class Config:
    description_tf_idf_name = "train_description_tf_idf.npz"
    top_index_save_name = "train_top_index.pk"
    result_name = "train.csv"
    handle_type = 0

    def __init__(self):
        pass


train_evaluate = {"description_tf_idf_name": "train_description_tf_idf.npz",
                  "top_index_save_name": "train_top_index.pk",
                  "result_name": "train.csv",
                  "handle_type": 0}

train_evaluate_shorter = {"paper_tf_idf_name": "paper_tf_idf_shorter.npz",
                          "description_tf_idf_name": "train_description_tf_idf_1.npz",
                          "validation_description_tf_idf_name": "train_description_tf_idf_1.npz",
                          "top_index_save_name": "train_top_index_1.pk",
                          "result_name": "train_shorter.csv",
                          "tokenizer_name": "tokenizer_shorter.pk",
                          "handle_type": 2
                          }
