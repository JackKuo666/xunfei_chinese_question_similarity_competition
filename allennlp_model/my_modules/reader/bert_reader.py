from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register("bert_reader")
class BertReader(DatasetReader):
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
        # 80% of the text_a length in the training set is less than 256, 512 - 256 = 256.
        tokens_a = self.tokenizer.tokenize(text_a)[:self.max_tokens//2]
        tokens_b = self.tokenizer.tokenize(text_b)[:self.max_tokens-len(tokens_a)]
        # 4、text_a+text_b 中间是sep 同时输入 bert

        tokens = self.tokenizer.add_special_tokens(tokens_a[1:-1], tokens_b[1:-1])

        text_field = TextField(tokens, self.token_indexers)

        fields = {"text": text_field}
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
