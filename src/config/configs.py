from dataclasses import dataclass


@dataclass()
class SaveInfoConfig:
    paper_embedding: str
    paper_id: str

    train_description_embedding: str
    train_description_id: str
    train_paper_id: str
    train_top_index: str
    train_prediction: str

    validation_description_embedding: str
    validation_description_id: str
    validation_top_index: str
    validation_prediction: str


@dataclass()
class TripletTrainConfig:
    batch_size: int = 16
    sample_number: int = 32
    random: bool = True
    hard: int = -1
    max_len: int = 100
    use_idf: bool = False
    use_self_train: bool = False
    embedding_size: int = 300


default_train_config = TripletTrainConfig()

triplet_config = SaveInfoConfig(paper_embedding="paper_info_triplet.pk",
                                paper_id="paper_id.pk",

                                train_description_embedding="train_description_triplet.pk",
                                train_paper_id="train_paper_id_of_description.pk",
                                train_description_id="train_description_id.pk",
                                train_top_index="top_index_triplet.pk",
                                train_prediction="train_triplet.csv",

                                validation_description_embedding="validation_description.pk",
                                validation_description_id="validation_id.pk",
                                validation_top_index="top_index_triplet_validation.pk",
                                validation_prediction="validation_triplet.csv",
                                )

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
