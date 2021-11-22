from app.pipelines.base import Pipeline, PipelineException  # isort:skip

from app.pipelines.feature_extraction import FeatureExtractionPipeline
from app.pipelines.question_answering import QuestionAnsweringPipeline
from app.pipelines.sentence_similarity import SentenceSimilarityPipeline
from app.pipelines.text_classification import TextClassificationPipeline
