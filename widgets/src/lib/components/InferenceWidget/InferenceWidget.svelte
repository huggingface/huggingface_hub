<script>
	import type { SvelteComponent } from "svelte";
	import type { PipelineType } from "../../interfaces/Types";
	import type { WidgetProps } from "./shared/types";

	import AudioClassificationWidget from "./widgets/AudioClassificationWidget/AudioClassificationWidget.svelte";
	import AudioToAudioWidget from "./widgets/AudioToAudioWidget/AudioToAudioWidget.svelte";
	import AutomaticSpeechRecognitionWidget from "./widgets/AutomaticSpeechRecognitionWidget/AutomaticSpeechRecognitionWidget.svelte";
	import ConversationalWidget from "./widgets/ConversationalWidget/ConversationalWidget.svelte";
	import FeatureExtractionWidget from "./widgets/FeatureExtractionWidget/FeatureExtractionWidget.svelte";
	import FillMaskWidget from "./widgets/FillMaskWidget/FillMaskWidget.svelte";
	import ImageClassificationWidget from "./widgets/ImageClassificationWidget/ImageClassificationWidget.svelte";
	import ImageSegmentationWidget from "./widgets/ImageSegmentationWidget/ImageSegmentationWidget.svelte";
	import ObjectDetectionWidget from "./widgets/ObjectDetectionWidget/ObjectDetectionWidget.svelte";
	import QuestionAnsweringWidget from "./widgets/QuestionAnsweringWidget/QuestionAnsweringWidget.svelte";
	import SentenceSimilarityWidget from "./widgets/SentenceSimilarityWidget/SentenceSimilarityWidget.svelte";
	import SummarizationWidget from "./widgets/SummarizationWidget/SummarizationWidget.svelte";
	import TableQuestionAnsweringWidget from "./widgets/TableQuestionAnsweringWidget/TableQuestionAnsweringWidget.svelte";
	import TextGenerationWidget from "./widgets/TextGenerationWidget/TextGenerationWidget.svelte";
	import TextToImageWidget from "./widgets/TextToImageWidget/TextToImageWidget.svelte";
	import TextToSpeechWidget from "./widgets/TextToSpeechWidget/TextToSpeechWidget.svelte";
	import TokenClassificationWidget from "./widgets/TokenClassificationWidget/TokenClassificationWidget.svelte";
	import StructuredDataClassificationWidget from "./widgets/StructuredDataClassificationWidget/StructuredDataClassificationWidget.svelte";
	import ZeroShotClassificationWidget from "./widgets/ZeroShowClassificationWidget/ZeroShotClassificationWidget.svelte";

	export let apiToken: WidgetProps["apiToken"] = undefined;
	export let callApiOnMount = false;
	export let apiUrl = "https://api-inference.huggingface.co";
	export let model: WidgetProps["model"];
	export let noTitle = false;
	export let shouldUpdateUrl = false;

	// Note: text2text-generation, text-generation and translation all
	// uses the TextGenerationWidget as they work almost the same.
	// Same goes for fill-mask and text-classification.
	// In the future it may be useful / easier to maintain if we created
	// a single dedicated widget for each pipeline type.
	const WIDGET_COMPONENTS: {
		[key in keyof typeof PipelineType]?: typeof SvelteComponent;
	} = {
		"audio-to-audio": AudioToAudioWidget,
		"audio-classification": AudioClassificationWidget,
		"automatic-speech-recognition": AutomaticSpeechRecognitionWidget,
		conversational: ConversationalWidget,
		"feature-extraction": FeatureExtractionWidget,
		"fill-mask": FillMaskWidget,
		"image-classification": ImageClassificationWidget,
		"image-segmentation": ImageSegmentationWidget,
		"object-detection": ObjectDetectionWidget,
		"question-answering": QuestionAnsweringWidget,
		"sentence-similarity": SentenceSimilarityWidget,
		summarization: SummarizationWidget,
		"table-question-answering": TableQuestionAnsweringWidget,
		"text2text-generation": TextGenerationWidget,
		"text-classification": FillMaskWidget,
		"text-generation": TextGenerationWidget,
		"token-classification": TokenClassificationWidget,
		"text-to-image": TextToImageWidget,
		"text-to-speech": TextToSpeechWidget,
		translation: TextGenerationWidget,
		"structured-data-classification": StructuredDataClassificationWidget,
		"zero-shot-classification": ZeroShotClassificationWidget,
	};

	$: widgetComponent = WIDGET_COMPONENTS[model.pipeline_tag ?? ""];

	// prettier-ignore
	$: widgetProps = ({
		apiToken,
		apiUrl,
		callApiOnMount,
		model,
		noTitle,
		shouldUpdateUrl,
	}) as WidgetProps;
</script>

{#if widgetComponent}
	<svelte:component
		this={WIDGET_COMPONENTS[model.pipeline_tag ?? ""]}
		{...widgetProps}
	/>
{/if}
