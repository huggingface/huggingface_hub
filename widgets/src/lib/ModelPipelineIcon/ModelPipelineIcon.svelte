<script lang="ts">
	import type { IconData } from "../Icon/Types";
	import type { PipelineType } from "../interfaces/Types";

	import Icon from "../Icon/Icon.svelte";
	import IconAudioClassification from "../Icon/data/IconAudioClassification";
	import IconAudioToAudio from "../Icon/data/IconAudioToAudio";
	import IconAutomaticSpeechRecognition from "../Icon/data/IconAutomaticSpeechRecognition";
	import IconConversational from "../Icon/data/IconConversational";
	import IconFeatureExtraction from "../Icon/data/IconFeatureExtraction";
	import IconFillMask from "../Icon/data/IconFillMask";
	import IconImageClassification from "../Icon/data/IconImageClassification";
	import IconImageSegmentation from "../Icon/data/IconImageSegmentation";
	import IconObjectDetection from "../Icon/data/IconObjectDetection";
	import IconQuestionAnswering from "../Icon/data/IconQuestionAnswering";
	import IconSentenceSimilarity from "../Icon/data/IconSentenceSimilarity";
	import IconStructuredDataClassification from "../Icon/data/IconStructuredDataClassification";
	import IconSummarization from "../Icon/data/IconSummarization";
	import IconTableQuestionAnswering from "../Icon/data/IconTableQuestionAnswering";
	import IconText2textGeneration from "../Icon/data/IconText2textGeneration";
	import IconTextClassification from "../Icon/data/IconTextClassification";
	import IconTextGeneration from "../Icon/data/IconTextGeneration";
	import IconTextToImage from "../Icon/data/IconTextToImage";
	import IconImageToText from "../Icon/data/IconImageToText";
	import IconTextToSpeech from "../Icon/data/IconTextToSpeech";
	import IconTokenClassification from "../Icon/data/IconTokenClassification";
	import IconTranslation from "../Icon/data/IconTranslation";
	import IconVoiceActivityDetection from "../Icon/data/IconVoiceActivityDetection";
	import IconZeroShotClassification from "../Icon/data/IconZeroShotClassification";

	export let classNames = "";
	export let pipeline = "";

	const ICON_DATA: {
		[key in keyof typeof PipelineType]?: IconData;
	} = {
		/// Keep same order as in huggingface_hub/Types.ts
		/// for easy mapping.
		"text-classification": IconTextClassification,
		"token-classification": IconTokenClassification,
		"table-question-answering": IconTableQuestionAnswering,
		"question-answering": IconQuestionAnswering,
		"zero-shot-classification": IconZeroShotClassification,
		"image-to-text": IconImageToText,
		translation: IconTranslation,
		summarization: IconSummarization,
		conversational: IconConversational,
		"feature-extraction": IconFeatureExtraction,
		"text-generation": IconTextGeneration,
		"text2text-generation": IconText2textGeneration,
		"fill-mask": IconFillMask,
		"sentence-similarity": IconSentenceSimilarity,
		"text-to-speech": IconTextToSpeech,
		"automatic-speech-recognition": IconAutomaticSpeechRecognition,
		"audio-to-audio": IconAudioToAudio,
		"voice-activity-detection": IconVoiceActivityDetection,
		"image-classification": IconImageClassification,
		"object-detection": IconObjectDetection,
		"image-segmentation": IconImageSegmentation,
		"text-to-image": IconTextToImage,
		"structured-data-classification": IconStructuredDataClassification,
		"audio-classification": IconAudioClassification,
	};
</script>

<Icon
	data={ICON_DATA[pipeline] ?? IconFillMask}
	{classNames}
/>
