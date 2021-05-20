from typing import Any, Dict, List

from app.pipelines import Pipeline
from flair.data import Sentence
from flair.models import SequenceTagger


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        self.tagger = SequenceTagger.load(model_id)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A substring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
        """
        sentence: Sentence = Sentence(inputs)

        # Also show scores for recognized NEs
        self.tagger.predict(sentence, all_tag_prob=True, label_name="predicted")

        entities = []
        for span in sentence.get_spans("predicted"):
            if len(span.tokens) == 0:
                continue
            current_entity = {
                "entity_group": span.tag,
                "word": span.to_original_text(),
                "start": span.tokens[0].start_pos,
                "end": span.tokens[-1].end_pos,
                "score": span.score,
            }

            entities.append(current_entity)

        return entities
