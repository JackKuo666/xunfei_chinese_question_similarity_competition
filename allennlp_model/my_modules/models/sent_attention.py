from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight_a = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_a.data.normal_(mean=0.0, std=0.05)

        self.bias_a = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias_a.data.copy_(torch.from_numpy(b))

        self.weight_b = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_b.data.normal_(mean=0.0, std=0.05)

        self.bias_b = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_b.data.copy_(torch.from_numpy(b))
        self.output_dim = hidden_size

    def forward(self, encoded_a, encoded_b):
        # 1、思路一：将两者分别乘以w,再加和，输出
        # Shape: (batch_size, encoding_dim)

        # linear
        encoded_a = torch.matmul(encoded_a, self.weight_a) + self.bias_a  # b x hidden
        encoded_b = torch.matmul(encoded_b, self.weight_b) + self.bias_b  # b x hidden
        # 1.sum
        batch_outputs = encoded_a + encoded_b
        # 2.拼接
        # batch_outputs = torch.cat([encoded_a, encoded_b], dim=-1)

        return batch_outputs

    def get_output_dim(self):
        return self.output_dim


@Model.register("sent_attention")
class SentAttention(Model):
    def __init__(
        self, vocab: Vocabulary,
            embedder_a: TextFieldEmbedder, encoder_a: Seq2VecEncoder,
            embedder_b: TextFieldEmbedder, encoder_b: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder_a = embedder_a
        self.encoder_a = encoder_a
        self.embedder_b = embedder_b
        self.encoder_b = encoder_b
        self.hidden_size = self.encoder_a.get_output_dim()
        self.sent_attention = Attention(self.hidden_size)

        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(self.sent_attention.get_output_dim(), num_labels)
        # self.classifier = torch.nn.Linear(self.hidden_size*2, num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text_a: TextFieldTensors, text_b: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # todo: 这里两个使用同一个参数，可以:
        #  1、使用相同的embedder,但是使用不同的encoder
        #  2、使用不同的embedder和不同的encoder      【doing】

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_a = self.embedder_a(text_a)
        # Shape: (batch_size, num_tokens)
        mask_a = util.get_text_field_mask(text_a)
        # Shape: (batch_size, encoding_dim)
        encoded_a = self.encoder_a(embedded_a, mask_a)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_b = self.embedder_b(text_b)
        # Shape: (batch_size, num_tokens)
        mask_b = util.get_text_field_mask(text_b)
        # Shape: (batch_size, encoding_dim)
        encoded_b = self.encoder_b(embedded_b, mask_b)

        # 1、Shape: (batch_size, encoding_dim)
        sent_hiddens = self.sent_attention(encoded_a, encoded_b)

        # 2、Shape: (batch_size, encoding_dim*2)
        # todo 直接拼接 为 (batch_size, encoding_dim*2)
        # sent_hiddens = torch.cat([encoded_a, encoded_b], dim=-1)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(sent_hiddens)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        label = torch.argmax(output_dict["probs"], dim=1)
        label = [self.vocab.get_token_from_index(int(i), "labels") for i in label]
        output_dict["label"] = label
        return output_dict
