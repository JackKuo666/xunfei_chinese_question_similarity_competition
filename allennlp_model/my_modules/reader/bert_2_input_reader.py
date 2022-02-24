from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register("bert_2_input_reader")
class SentAttentionReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text_a: str, text_b: str, label: str = None) -> Instance:
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        if self.max_tokens:
            tokens_a = tokens_a[: self.max_tokens]
            tokens_b = tokens_b[: self.max_tokens]
        text_field_a = TextField(tokens_a, self.token_indexers)
        text_field_b = TextField(tokens_b, self.token_indexers)
        # 4、同时加入两个就是两个inputs text了
        fields = {"text_a": text_field_a, "text_b": text_field_b}
        # 3、如果含有label的话，就加入，说明是训练；没有的话不加入，说明是测试
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as lines:
            for line in lines:
                line = line.strip().split("\t")
                # 1、这里判断输入的train(含有label)还是test(不含label)
                if len(line) == 3:
                    text_a, text_b, categories = line
                else:
                    text_a, text_b, = line
                    categories = None
                if text_a == "text_a":
                    continue
                yield self.text_to_instance(text_a, text_b, categories)
