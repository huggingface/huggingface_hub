
/**
 * Add your new library here.
 */
export enum ModelLibrary {
	'asteroid'               = 'Asteroid',
	'espnet'                 = 'ESPnet',
	'flair'                  = 'Flair',
	'pyannote'               = 'Pyannote',
	'sentence-transformers'  = 'Sentence Transformers',
	'spacy'                  = 'spaCy',
	'tensorflowtts'          = 'TensorFlowTTS',
	'timm'                   = 'Timm',
	'transformers'           = 'Transformers',
};

export const ALL_MODEL_LIBRARY_KEYS = Object.keys(ModelLibrary) as (keyof typeof ModelLibrary)[];

/**
 * subset of model metadata that
 * a code snippet can depend on.
 */
interface ModelData {
	/**
	 * id of model (e.g. 'user/repo_name')
	 */
	modelId: string;
	/**
	 * name of repository.
	 */
	repoName: string;
	/**
	 * is this model private?
	 */
	private: boolean;
	/**
	 * all the model tags
	 */
	tags: string[];
	/**
	 * this is transformers-specific
	 */
	autoArchitecture: string;
}


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

//#region snippets

const asteroid = (model: ModelData) =>
`from asteroid.models import BaseModel
  
model = BaseModel.from_pretrained("${model.modelId}")`;

const espnetTTS = (model: ModelData) =>
`from espnet2.bin.tts_inference import Text2Speech
    
model = Text2Speech.from_pretrained("${model.modelId}")

speech, *_ = model("text to generate speech from")`;

const espnetASR = (model: ModelData) =>
`from espnet2.bin.asr_inference import Speech2Text
    
model = Speech2Text.from_pretrained(
  "${model.modelId}"
)

speech, rate = soundfile.read("speech.wav")
text, *_ = model(speech)`;

const espnetUnknown = () =>
`unknown model type (must be text-to-speech or automatic-speech-recognition)`;

const espnet = (model: ModelData) => {
	if (model.tags.includes("text-to-speech")){
		return espnetTTS(model);
	} else if (model.tags.includes("automatic-speech-recognition")) {
		return espnetASR(model);
	}
	return espnetUnknown();
};

const flair = (model: ModelData) =>
`from flair.models import SequenceTagger
  
tagger = SequenceTagger.load("${model.modelId}")`;

const pyannote = (model: ModelData) =>
`from pyannote.audio.core.inference import Inference
  
model = Inference("${model.modelId}")

# inference on the whole file
model("file.wav")

# inference on an excerpt
from pyannote.core import Segment
excerpt = Segment(start=2.0, end=5.0)
model.crop("file.wav", excerpt)`;

const tensorflowttsTextToMel = (model: ModelData) =>
`from tensorflow_tts.inference import AutoProcessor, TFAutoModel

processor = AutoProcessor.from_pretrained("${model.modelId}")
model = TFAutoModel.from_pretrained("${model.modelId}")
`;

const tensorflowttsMelToWav = (model: ModelData) =>
`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${model.modelId}")
audios = model.inference(mels)
`;

const tensorflowttsUnknown = (model: ModelData) =>
`from tensorflow_tts.inference import TFAutoModel

model = TFAutoModel.from_pretrained("${model.modelId}")
`;

const tensorflowtts = (model: ModelData) => {
	if (model.tags.includes("text-to-mel")){
		return tensorflowttsTextToMel(model);
	} else if (model.tags.includes("mel-to-wav")) {
		return tensorflowttsMelToWav(model);
	}
	return tensorflowttsUnknown(model);
};

const timm = (model: ModelData) =>
`import timm

model = timm.create_model("${model.modelId}", pretrained=True)`;

const sentenceTransformers = (model: ModelData) =>
`from sentence_transformers import SentenceTransformer

model = SentenceTransformer("${model.modelId}")`;

const spacy = (model: ModelData) =>
`# pip install https://huggingface.co/${model.modelId}/resolve/main/${model.repoName}.whl

# Importing as module.
import ${model.repoName}
nlp = ${model.repoName}.load()

# Using spaCy.load().
import spacy
nlp = spacy.load(${model.repoName})`;

const transformers = (model: ModelData) =>
`from transformers import AutoTokenizer, ${model.autoArchitecture}
  
tokenizer = AutoTokenizer.from_pretrained("${model.modelId}"${model.private ? `, use_auth_token=True` : ``})

model = ${model.autoArchitecture}.from_pretrained("${model.modelId}"${model.private ? `, use_auth_token=True` : ``})`;

//#endregion



export const MODEL_LIBRARIES_UI_ELEMENTS: { [key in keyof typeof ModelLibrary]: LibraryUiElement } = {
	asteroid: {
		btnLabel: "Asteroid",
		repoName: "Asteroid",
		repoUrl: "https://github.com/asteroid-team/asteroid",
		snippet: asteroid,
	},
	espnet: {
		btnLabel: "ESPnet",
		repoName: "ESPnet",
		repoUrl: "https://github.com/espnet/espnet",
		snippet: espnet,
	},
	flair: {
		btnLabel: "Flair",
		repoName: "Flair",
		repoUrl: "https://github.com/flairNLP/flair",
		snippet: flair,
	},
	pyannote: {
		btnLabel: "pyannote",
		repoName: "pyannote-audio",
		repoUrl: "https://github.com/pyannote/pyannote-audio",
		snippet: pyannote,
	},
	"sentence-transformers": {
		btnLabel: "sentence-transformers",
		repoName: "sentence-transformers",
		repoUrl: "https://github.com/UKPLab/sentence-transformers",
		snippet: sentenceTransformers,
	},
	spacy: {
		btnLabel: "spacy",
		repoName: "spacy",
		repoUrl: "https://github.com/explosion/spaCy/stargazers",
		snippet: spacy,
	},
	tensorflowtts : {
		btnLabel: "TensorFlowTTS",
		repoName: "TensorFlowTTS",
		repoUrl: "https://github.com/TensorSpeech/TensorFlowTTS",
		snippet: tensorflowtts
	},
	timm: {
		btnLabel: "timm",
		repoName: "pytorch-image-models",
		repoUrl: "https://github.com/rwightman/pytorch-image-models",
		snippet: timm,
	},
	transformers: {
		btnLabel: "Transformers",
		repoName: "🤗/transformers",
		repoUrl: "https://github.com/huggingface/transformers",
		snippet: transformers,
	},
} as const;

