from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
from allennlp.common.util import JsonDict, sanitize


@Predictor.register("sentence_sim_classifier")
class SentenceClassifierPredictor(Predictor):

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text_a = json_dict["text_a"]
        text_b = json_dict["text_b"]
        return self._dataset_reader.text_to_instance(text_a, text_b)

    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        self.line = line
        line = line.strip().split("\t")
        text_a, text_b = line
        return {"text_a": text_a, "text_b": text_b}

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return str(outputs["label"]) + "\n"


    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        # print("inputs", inputs)
        outputs = [{"label": i["label"]} for i in outputs]
        # print("outputs", outputs)

        return outputs