import type { ModelData } from "./Types";

/**
 * Add your new library here.
 */
export enum ModelLibrary {
	"adapter-transformers"   = "Adapter Transformers",
	"allennlp"               = "allenNLP",
	"asteroid"               = "Asteroid",
	"espnet"                 = "ESPnet",
	"fairseq"                = "Fairseq",
	"flair"                  = "Flair",
	"keras"                  = "Keras",
	"pyannote"               = "Pyannote",
	"sentence-transformers"  = "Sentence Transformers",
	"sklearn"                = "Scikit-learn",
	"spacy"                  = "spaCy",
	"speechbrain"            = "speechbrain",
	"tensorflowtts"          = "TensorFlowTTS",
	"timm"                   = "Timm",
	"fastai"                 = "fastai",
	"transformers"           = "Transformers",
	"stanza"                 = "Stanza",
	"fasttext"               = "fastText",
	"stable-baselines3"      = "Stable-Baselines3",
}

export const ALL_MODEL_LIBRARY_KEYS = Object.keys(ModelLibrary) as (keyof typeof ModelLibrary)[];


/**
 * Elements configurable by a model library.
 */
export interface LibraryUiElement {
	/**
	 * Name displayed on the main
	 * call-to-action button on the model page.
	 */
	btnLabel:  string;
	/**
	 * Repo name
	 */
	repoName: string;
	/**
	 * URL to library's repo
	 */
	repoUrl:   string;
	/**
	 * Code snippet displayed on model page
	 */
	snippet:   (model: ModelData) => string;
}

function nameWithoutNamespace(modelId: string): string {
	const splitted = modelId.split("/");
	return splitted.length === 1 ? splitted[0] : splitted[1];
}

//#region snippets

const adapter_transformers = (model: ModelData) =>
	`from transformers import ${model.config?.adapter_transformers?.model_class}

model = ${model.config?.adapter_transformers?.model_class}.from_pretrained("${model.config?.adapter_transformers?.model_name}")
model.load_adapter("${model.id}", source="hf")`;

const allennlpUnknown = (model: ModelData) =>
	`import allennlp_models
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("hf://${model.id}")`;

const allennlpQuestionAnswering = (model: ModelData) =>
	`import allennlp_models
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("hf://${model.id}")
predictor_input = {"passage": "My name is Wolfgang and I live in Berlin", "question": "Where do I live?"}
predictions = predictor.predict_json(predictor_input)`;

const allennlp = (model: ModelData) => {
	if (model.tags?.includes("question-answering")) {
		return allennlpQuestionAnswering(model);
	}
	return allennlpUnknown(model);
};

const asteroid = (model: ModelData) =>
	`from asteroid.models import BaseModel
  
model = BaseModel.from_pretrained("${model.id}")`;

const espnetTTS = (model: ModelData) =>
	`from espnet2.bin.tts_inference import Text2Speech
    
model = Text2Speech.from_pretrained("${model.id}")

speech, *_ = model("text to generate speech from")`;

const espnetASR = (model: ModelData) =>
	`from espnet2.bin.asr_inference import Speech2Text
    
model = Speech2Text.from_pretrained(
  "${model.id}"
)

speech, rate = soundfile.read("speech.wav")
text, *_ = model(speech)`;

const espnetUnknown = () =>
	`unknown model type (must be text-to-speech or automatic-speech-recognition)`;

const espnet = (model: ModelData) => {
	if (model.tags?.includes("text-to-speech")) {
		return espnetTTS(model);
	} else if (model.tags?.includes("automatic-speech-recognition")) {
		return espnetASR(model);
	}
	return espnetUnknown();
};

const fairseq = (model: ModelData) =>
	`from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "${model.id}"
)`;


const flair = (model: ModelData) =>
	`from flair.models import SequenceTagger
  
tagger = SequenceTagger.load("${model.id}")`;

const keras = (model: ModelData) =>
	`from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("${model.id}")
`;

const pyannote = (model: ModelData) =>
	`from pyannote.audio.core.inference import Inference
  
model = Inference("${model.id}")

# inference on the whole file
model("file.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)
model.crop("file.wav", excerpt)`;

const tensorflowttsTextToMel = (model: ModelData) =>
	`from tensorflow_tts.inference import AutoProcessor, TFAutoModel

processor = AutoProcessor.from_pretrained("${model.id}")
model = TFAutoModel.from_pretrained("${model.id}")
`;

const tensorflowttsMelToWav = (model: ModelData) =>
	`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${model.id}")
audios = model.inference(mels)
`;

const tensorflowttsUnknown = (model: ModelData) =>
	`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${model.id}")
`;

const tensorflowtts = (model: ModelData) => {
	if (model.tags?.includes("text-to-mel")) {
		return tensorflowttsTextToMel(model);
	} else if (model.tags?.includes("mel-to-wav")) {
		return tensorflowttsMelToWav(model);
	}
	return tensorflowttsUnknown(model);
};

const timm = (model: ModelData) =>
	`import timm

model = timm.create_model("hf_hub:${model.id}", pretrained=True)`;

const sklearn = (model: ModelData) =>
	`from huggingface_hub import hf_hub_download
import joblib

model = joblib.load(
	hf_hub_download("${model.id}", "sklearn_model.joblib")
)`;

const fastai = (model: ModelData) =>
	`from huggingface_hub import hf_hub_download
from fastai.learner import load_learner

model = load_learner(
    hf_hub_download("${model.id}", "model.pkl")
)`;

const sentenceTransformers = (model: ModelData) =>
	`from sentence_transformers import SentenceTransformer

model = SentenceTransformer("${model.id}")`;

const spacy = (model: ModelData) =>
	`!pip install https://huggingface.co/${model.id}/resolve/main/${nameWithoutNamespace(model.id)}-any-py3-none-any.whl

# Using spacy.load().
import spacy
nlp = spacy.load("${nameWithoutNamespace(model.id)}")

# Importing as module.
import ${nameWithoutNamespace(model.id)}
nlp = ${nameWithoutNamespace(model.id)}.load()`;

const speechBrainMethod = (speechbrainInterface: string) => {
	switch (speechbrainInterface) {
		case "EncoderClassifier":
		   return "classify_file";
		case "EncoderDecoderASR":
		case "EncoderASR":
			return "transcribe_file";
		case "SpectralMaskEnhancement":
			return "enhance_file";
		case "SepformerSeparation":
			return "separate_file";
		default:
			return undefined;
	}
};

const speechbrain = (model: ModelData) => {
	const speechbrainInterface = model.config?.speechbrain?.interface;
	if (speechbrainInterface === undefined) {
		return `# interface not specified in config.json`;
	}

	const speechbrainMethod = speechBrainMethod(speechbrainInterface);
	if (speechbrainMethod === undefined) {
		return `# interface in config.json invalid`;
	}

	return `from speechbrain.pretrained import ${speechbrainInterface}
model = ${speechbrainInterface}.from_hparams(
  "${model.id}"
)
model.${speechbrainMethod}("file.wav")`;
};

const transformers = (model: ModelData) => {
	const info = model.transformersInfo;
	if (!info) {
		return `# ⚠️ Type of model unknown`;
	}
	if (info.processor) {
		const varName = info.processor === "AutoTokenizer" ? "tokenizer"
			: info.processor === "AutoFeatureExtractor" ? "extractor"
				: "processor"
		;
		return [
			`from transformers import ${info.processor}, ${info.auto_model}`,
			"",
			`${varName} = ${info.processor}.from_pretrained("${model.id}"${model.private ? ", use_auth_token=True" : ""})`,
			"",
			`model = ${info.auto_model}.from_pretrained("${model.id}"${model.private ? ", use_auth_token=True" : ""})`,
		].join("\n");
	} else {
		return [
			`from transformers import ${info.auto_model}`,
			"",
			`model = ${info.auto_model}.from_pretrained("${model.id}"${model.private ? ", use_auth_token=True" : ""})`,
		].join("\n");
	}
};

const fasttext = (model: ModelData) =>
	`from huggingface_hub import hf_hub_download
import fasttext

model = fasttext.load_model(hf_hub_download("${model.id}", "model.bin"))`;

const stableBaselines3 = (model: ModelData) =>
	`from huggingface_sb3 import load_from_hub
	checkpoint = load_from_hub(
    		repo_id="${model.id}",
    		filename="{MODEL FILENAME}",
	)`;

//#endregion



export const MODEL_LIBRARIES_UI_ELEMENTS: { [key in keyof typeof ModelLibrary]?: LibraryUiElement } = {
	// ^^ TODO(remove the optional ? marker when Stanza snippet is available)
	"adapter-transformers": {
		btnLabel: "Adapter Transformers",
		repoName: "adapter-transformers",
		repoUrl:  "https://github.com/Adapter-Hub/adapter-transformers",
		snippet:  adapter_transformers,
	},
	"allennlp": {
		btnLabel: "AllenNLP",
		repoName: "AllenNLP",
		repoUrl:  "https://github.com/allenai/allennlp",
		snippet:  allennlp,
	},
	"asteroid": {
		btnLabel: "Asteroid",
		repoName: "Asteroid",
		repoUrl:  "https://github.com/asteroid-team/asteroid",
		snippet:  asteroid,
	},
	"espnet": {
		btnLabel: "ESPnet",
		repoName: "ESPnet",
		repoUrl:  "https://github.com/espnet/espnet",
		snippet:  espnet,
	},
	"fairseq": {
		btnLabel: "Fairseq",
		repoName: "fairseq",
		repoUrl:  "https://github.com/pytorch/fairseq",
		snippet:  fairseq,
	},
	"flair": {
		btnLabel: "Flair",
		repoName: "Flair",
		repoUrl:  "https://github.com/flairNLP/flair",
		snippet:  flair,
	},
	"keras": {
		btnLabel: "Keras",
		repoName: "Keras",
		repoUrl:  "https://github.com/keras-team/keras",
		snippet:  keras,
	},
	"pyannote": {
		btnLabel: "pyannote",
		repoName: "pyannote-audio",
		repoUrl:  "https://github.com/pyannote/pyannote-audio",
		snippet:  pyannote,
	},
	"sentence-transformers": {
		btnLabel: "sentence-transformers",
		repoName: "sentence-transformers",
		repoUrl:  "https://github.com/UKPLab/sentence-transformers",
		snippet:  sentenceTransformers,
	},
	"sklearn": {
		btnLabel: "Scikit-learn",
		repoName: "Scikit-learn",
		repoUrl:  "https://github.com/scikit-learn/scikit-learn",
		snippet:  sklearn,
	},
	"fastai": {
		btnLabel: "fastai",
		repoName: "fastai",
		repoUrl:  "https://github.com/fastai/fastai",
		snippet:  fastai,
	},
	"spacy": {
		btnLabel: "spaCy",
		repoName: "spaCy",
		repoUrl:  "https://github.com/explosion/spaCy",
		snippet:  spacy,
	},
	"speechbrain": {
		btnLabel: "speechbrain",
		repoName: "speechbrain",
		repoUrl:  "https://github.com/speechbrain/speechbrain",
		snippet:  speechbrain,
	},
	"tensorflowtts": {
		btnLabel: "TensorFlowTTS",
		repoName: "TensorFlowTTS",
		repoUrl:  "https://github.com/TensorSpeech/TensorFlowTTS",
		snippet:  tensorflowtts,
	},
	"timm": {
		btnLabel: "timm",
		repoName: "pytorch-image-models",
		repoUrl:  "https://github.com/rwightman/pytorch-image-models",
		snippet:  timm,
	},
	"transformers": {
		btnLabel: "Transformers",
		repoName: "🤗/transformers",
		repoUrl:  "https://github.com/huggingface/transformers",
		snippet:  transformers,
	},
	"fasttext": {
		btnLabel: "fastText",
		repoName: "fastText",
		repoUrl:  "https://fasttext.cc/",
		snippet:  fasttext,
	},
	"stable-baselines3": {
		btnLabel: "stable-baselines3",
		repoName: "stable-baselines3",
		repoUrl:  "https://github.com/huggingface/huggingface_sb3",
		snippet:  stableBaselines3,
	},
} as const;

