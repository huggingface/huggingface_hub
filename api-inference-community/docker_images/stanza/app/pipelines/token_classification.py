import os
from typing import Any, Dict, List

import stanza
from app.pipelines import Pipeline
from stanza import Pipeline as pipeline


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        namespace, model_name = model_id.split("/")

        path = os.path.join(
            os.environ.get("HUGGINGFACE_HUB_CACHE", "."), namespace, model_name
        )

        lang = model_name.replace("stanza-", "")
        stanza.download(model_dir=path, lang=lang)
        self.model = pipeline(model_dir=path, lang=lang)

    def __call__(self, inputs: str) -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`str`):
                a string containing some text
        Return:
            A :obj:`list`:. The object returned should be like [{"entity_group": "XXX", "word": "some word", "start": 3, "end": 6, "score": 0.82}] containing :
                - "entity_group": A string representing what the entity is.
                - "word": A rubstring of the original string that was detected as an entity.
                - "start": the offset within `input` leading to `answer`. context[start:stop] == word
                - "end": the ending offset within `input` leading to `answer`. context[start:stop] === word
                - "score": A score between 0 and 1 describing how confident the model is for this entity.
        """
        doc = self.model(inputs)

        entities = []
        if "ner_model_path" in self.model.config.keys():

            for entity in doc.entities:
                entity_dict = {
                    "entity_group": entity.type,
                    "word": entity.text,
                    "start": entity.start_char,
                    "end": entity.end_char,
                    "score": 1.0,
                }
                entities.append(entity_dict)
            
        else:

            for sent in doc.sentences:
                for entity in sent.words:
                    
                    entity_dict = {
                        "entity_group": entity.upos,
                        "word": entity.text,
                        "start": entity.start_char,
                        "end": entity.end_char,
                        "score": 1.0,
                    }

                    entities.append(entity_dict)

        return entities
