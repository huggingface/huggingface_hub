from app.pipelines.base import Pipeline, PipelineException  # isort:skip

from app.pipelines.audio_to_audio import AudioToAudioPipeline
from app.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from app.pipelines.image_classification import ImageClassificationPipeline
from app.pipelines.question_answering import QuestionAnsweringPipeline
from app.pipelines.text_to_speech import TextToSpeechPipeline
from app.pipelines.token_classification import TokenClassificationPipeline
