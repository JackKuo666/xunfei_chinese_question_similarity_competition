{
    "dataset_reader" : {
        "type": "bert_reader",
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
    "model": {
        "type": "bert_sent",
        "pretrained_model": "chinese_question_bert_base/",
        "requires_grad": true,
        "dropout": 0.2
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys":["text"],
            "batch_size": 16
        },
        "cuda_device": 0,
        "max_instances_in_memory": 1600
    },
    "trainer": {
        "validation_metric": ["+accuracy","-loss"],
        "patience": 2,
        "callbacks":["tensorboard"],
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-4
        },
        "num_epochs": 5
    }
}
