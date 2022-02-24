{
    "dataset_reader" : {
        "type": "bert_2_input_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "chinese_question_bert_base/"
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": "chinese_question_bert_base/"
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/chinese_question_sim/train_split_data/train_data.csv",
    "validation_data_path": "data/chinese_question_sim/train_split_data/dev_data.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab.tar.gz"
    },
    "model": {
        "type": "sent_attention",
        "embedder_a": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": "chinese_question_bert_base/"
                }
            }
        },
        "encoder_a": {
            "type": "bert_pooler",
            "pretrained_model": "chinese_question_bert_base/",
            "requires_grad": true,
            "dropout": 0.1
        },
        "embedder_b": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": "chinese_question_bert_base/"
                }
            }
        },
        "encoder_b": {
            "type": "bert_pooler",
            "pretrained_model": "chinese_question_bert_base/",
            "requires_grad": true,
            "dropout": 0.1
        }
    },
    "data_loader": {
        "batch_size": 8,
        "cuda_device": 0,
        "max_instances_in_memory": 1600,
        "shuffle": true
    },
    "trainer": {
        "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"checkpoint",
            "save_every_num_seconds": 1200
        },
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 1,
          "num_steps_per_epoch": 3088,
          "cut_frac": 0.06
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5,
            "weight_decay": 0.1
        },
        "num_epochs": 1
    }
}
