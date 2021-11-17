<script>
	import type { WidgetProps } from "../../shared/types";

	import { onMount } from "svelte";
	import WidgetOuputTokens from "../../shared/WidgetOutputTokens/WidgetOutputTokens.svelte";
	import WidgetTextarea from "../../shared/WidgetTextarea/WidgetTextarea.svelte";
	import WidgetSubmitBtn from "../../shared/WidgetSubmitBtn/WidgetSubmitBtn.svelte";
	import WidgetWrapper from "../../shared/WidgetWrapper/WidgetWrapper.svelte";
	import {
		addInferenceParameters,
		getDemoInputs,
		getResponse,
		getSearchParams,
		updateUrl,
	} from "../../shared/helpers";

	interface EntityGroup {
		entity_group: string;
		score: number;
		word: string;
		start?: number;
		end?: number;
	}

	interface Span {
		end: number;
		index?: string;
		start: number;
		type: string;
	}

	export let apiToken: WidgetProps["apiToken"];
	export let apiUrl: WidgetProps["apiUrl"];
	export let callApiOnMount: WidgetProps["callApiOnMount"];
	export let model: WidgetProps["model"];
	export let noTitle: WidgetProps["noTitle"];
	export let shouldUpdateUrl: WidgetProps["shouldUpdateUrl"];

	let computeTime = "";
	let error: string = "";
	let isLoading = false;
	let modelLoading = {
		isLoading: false,
		estimatedTime: 0,
	};
	let output: Span[] = [];
	let outputJson: string;
	let text = "";
	let outputText = "";

	onMount(() => {
		const [textParam] = getSearchParams(["text"]);
		if (textParam) {
			text = textParam;
			getOutput();
		} else {
			const [demoText] = getDemoInputs(model, ["text"]);
			text = (demoText as string) ?? "";
			if (text && callApiOnMount) {
				getOutput();
			}
		}
	});

	async function getOutput(withModelLoading = false) {
		const trimmedText = text.trim();

		if (!trimmedText) {
			error = "You need to input some text";
			output = [];
			outputJson = "";
			return;
		}

		if (shouldUpdateUrl) {
			updateUrl({ text: trimmedText });
		}

		const requestBody = { inputs: trimmedText };
		addInferenceParameters(requestBody, model);

		isLoading = true;

		const res = await getResponse(
			apiUrl,
			model.id,
			requestBody,
			apiToken,
			parseOutput,
			withModelLoading
		);

		isLoading = false;
		// Reset values
		computeTime = "";
		error = "";
		modelLoading = { isLoading: false, estimatedTime: 0 };
		output = [];
		outputJson = "";

		if (res.status === "success") {
			computeTime = res.computeTime;
			output = res.output;
			outputJson = res.outputJson;
			outputText = text;
		} else if (res.status === "loading-model") {
			modelLoading = {
				isLoading: true,
				estimatedTime: res.estimatedTime,
			};
			getOutput(true);
		} else if (res.status === "error") {
			error = res.error;
		}
	}

	function isValidOutput(arg: any): arg is EntityGroup[] {
		return (
			Array.isArray(arg) &&
			arg.every((x) => {
				return (
					typeof x.word === "string" &&
					typeof x.entity_group === "string" &&
					typeof x.score === "number"
				);
			})
		);
	}

	function parseOutput(body: unknown): Span[] {
		if (isValidOutput(body)) {
			// Filter out duplicates
			const filteredEntries = body.reduce<EntityGroup[]>((acc, entry) => {
				const exists = acc.some((accEntry) =>
					Object.keys(entry).every((k) => entry[k] === accEntry[k])
				);
				return exists ? acc : [...acc, entry];
			}, []);

			const spans = filteredEntries.reduce<Span[]>((acc, entry) => {
				const span = getSpanData(entry, acc, text);
				return span ? [...acc, span] : acc;
			}, []);

			spans.sort((a, b) => {
				/// `a` should come first when the result is < 0
				return a.start === b.start
					? b.end - a.end /// CAUTION.
					: a.start - b.start;
			});

			// Check existence of **strict overlapping**
			spans.forEach((s, i) => {
				if (i < spans.length - 1) {
					const sNext = spans[i + 1];
					if (s.start < sNext.start && s.end > sNext.start) {
						console.warn("ERROR", "Spans: strict overlapping");
					}
				}
			});

			return spans;
		}
		throw new TypeError(
			"Invalid output: output must be of type Array<word:string; entity_group:string; score:number>"
		);
	}

	function getSpanData(
		entityGroup: EntityGroup,
		spans: Span[],
		text: string
	): Span | null {
		// When the API returns start/end information
		if (entityGroup.start && entityGroup.end) {
			const span = {
				type: entityGroup.entity_group,
				start: entityGroup.start,
				end: entityGroup.end,
			};
			return !spans.some((x) => equals(x, span)) ? span : null;
		}

		// This is a fallback when the API doesn't return
		// start/end information (when using python tokenizers for instance).
		const normalizedText = text.toLowerCase();

		let needle = entityGroup.word.toLowerCase();
		let idx = 0;
		while (idx !== -1) {
			idx = normalizedText.indexOf(needle, idx);
			if (idx === -1) {
				break;
			}
			const span: Span = {
				type: entityGroup.entity_group,
				start: idx,
				end: idx + needle.length,
			};
			if (!spans.some((x) => equals(x, span))) {
				return span;
			}
			idx++;
		}

		// Fix for incorrect detokenization in this pipeline.
		// e.g. John - Claude
		// todo: Fix upstream.
		needle = entityGroup.word.toLowerCase().replace(/ /g, "");
		idx = 0;
		while (idx !== -1) {
			idx = normalizedText.indexOf(needle, idx);
			if (idx === -1) {
				break;
			}
			const span: Span = {
				type: entityGroup.entity_group,
				start: idx,
				end: idx + needle.length,
			};
			if (!spans.some((x) => equals(x, span))) {
				return span;
			}
		}
		return null;
	}

	function equals(a: Span, b: Span): boolean {
		return a.type === b.type && a.start === b.start && a.end === b.end;
	}

	function previewInputSample(sample: Record<string, any>) {
		text = sample.text;
	}

	function applyInputSample(sample: Record<string, any>) {
		text = sample.text;
		getOutput();
	}
</script>

<WidgetWrapper
	{apiUrl}
	{applyInputSample}
	{computeTime}
	{error}
	{isLoading}
	{model}
	{modelLoading}
	{noTitle}
	{outputJson}
	{previewInputSample}
>
	<svelte:fragment slot="top">
		<form>
			<WidgetTextarea bind:value={text} />
			<WidgetSubmitBtn
				classNames="mt-2"
				{isLoading}
				onClick={() => {
					getOutput();
				}}
			/>
		</form>
	</svelte:fragment>
	<svelte:fragment slot="bottom">
		<WidgetOuputTokens classNames="mt-2" {output} text={outputText} />
	</svelte:fragment>
</WidgetWrapper>
