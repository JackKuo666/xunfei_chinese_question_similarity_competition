{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "/home/featurize/data/robert_pretrain/"
        },
        "token_indexers": {
            "robert": {
                "type": "pretrained_transformer",
                "model_name": "/home/featurize/data/robert_pretrain/"
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/chinese_question_sim/train_split_data/train_data.csv",
    "validation_data_path": "data/chinese_question_sim/train_split_data/dev_data.csv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "robert": {
                    "type": "pretrained_transformer",
                    "model_name": "/home/featurize/data/robert_pretrain/"
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "/home/featurize/data/robert_pretrain/",
            "requires_grad": true,
            "dropout": 0.1
        }
    },
    "data_loader": {
        "batch_size": 8,
        "cuda_device": 0,
        "max_instances_in_memory": 400,
        "shuffle": true
    },
    "trainer": {
        "num_epochs": 1,
        "validation_metric": "+accuracy",
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
        }
    }
}
