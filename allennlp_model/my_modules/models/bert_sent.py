from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np


# build sent encoder
class SentEncoder(torch.nn.Module):
    def __init__(self, sent_rep_size, sent_hidden_size=256, sent_num_layers=2, dropout=0.15):
        super(SentEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.sent_lstm = torch.nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len

        masked_outputs = outputs.masked_fill((~ batch_masks).bool(), float(-1e32))

        attn_scores = torch.nn.functional.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((~ batch_masks).bool(), 0.0)

        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


@Model.register("bert_sent")
class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary,
            pretrained_model: str,
            requires_grad: bool = True,
            dropout: float = 0.0
    ):
        super().__init__(vocab)

        from allennlp.common import cached_transformers

        model = cached_transformers.get(
            pretrained_model,
            False
        )

        self._dropout = torch.nn.Dropout(p=dropout)

        import copy

        self.model = copy.deepcopy(model)

        for param in self.model.embeddings.parameters():
            param.requires_grad = requires_grad
        for param in self.model.encoder.parameters():
            param.requires_grad = requires_grad
        for param in self.model.pooler.parameters():
            param.requires_grad = requires_grad

        self._embedding_dim = model.config.hidden_size

        self.sent_attention = Attention(self._embedding_dim)
        self.sent_encoder = SentEncoder(sent_rep_size=self._embedding_dim, sent_hidden_size=self._embedding_dim,
                                        sent_num_layers=2, dropout=0.15)

        num_labels = vocab.get_vocab_size("labels")
        # pooler 用
        # self.classifier = torch.nn.Linear(self._embedding_dim, num_labels)
        # cat 用
        self.classifier = torch.nn.Linear(self._embedding_dim*2, num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        # 6、使用 bert embedding + lstm + sent attention
        output = self.model.embeddings(text["bert"]["token_ids"], text["bert"]["type_ids"])
        encoded_text, attn_scores = self.sent_attention(output, text["bert"]["mask"])

        # encoded_text = self.sent_encoder(output, text["bert"]["mask"])
        #
        # encoded_text = torch.mean(encoded_text, dim=1)
        logits = self.classifier(encoded_text)

        # # 5、使用 最后一层 lstm + sent attention
        # output = self.model.embeddings(text["bert"]["token_ids"], text["bert"]["mask"], text["bert"]["type_ids"])
        # encoded_text = self.sent_encoder(output.last_hidden_state, text["bert"]["mask"])
        # encoded_text = torch.mean(encoded_text, dim=1)
        # logits = self.classifier(encoded_text)

        # # 4、使用 最后一层 sent attention
        # encoded_text, attn_scores = self.sent_attention(output.last_hidden_state, text["bert"]["mask"])
        # logits = self.classifier(encoded_text)

        # # 3、使用 最后一层取平均
        # encoded_text = torch.mean(output.last_hidden_state, dim=1)
        # logits = self.classifier(encoded_text)

        # # 2、使用 最后一层取平均 然后与 pooler 拼接
        # encoded_text = torch.mean(output.last_hidden_state, dim=1)
        # encoded_text = torch.cat([encoded_text, output.pooler_output], dim=-1)
        # logits = self.classifier(encoded_text)

        # 1、使用pooler
        # logits = self.classifier(output.pooler_output)
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
