from typing import Any, Dict, List

from app.pipelines import Pipeline


class TokenClassificationPipeline(Pipeline):
    def __init__(
        self,
        model_id: str,
    ):
        import stanza
        from stanza import Pipeline as pipeline

        full_model_path = model_id.split("/")

        if len(full_model_path) != 5:
            raise ValueError(
                f"Invalid model_id: {model_id}. It should have a namespace (:namespace:/:model_name:)"
            )
        model_name = full_model_path[-1]  # conll03.pt
        model_type = full_model_path[-2]  # ner
        namespace = full_model_path[0]  # stanfordnlp
        repo_id = full_model_path[1]  # stanza-en

        model_dir = f"https://huggingface.co/{namespace}/{repo_id}/resolve/main/{model_type}/{model_name}"

        stanza.download(model_dir=model_dir)
        model = pipeline(model_dir=model_dir)
        self.model = model

        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here

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

        entity_list = []

        for entity in doc.entities:

            entity_dict = {
                "entity_group": entity.type,
                "word": entity.text,
                "start": entity.start_char,
                "end": entity.end_char,
                "score": 1.0,
            }
            entity_list.append(entity_dict)
        return entity_list
