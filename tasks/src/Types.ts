import type { ModelLibrary } from "../../js/src/lib/interfaces/Libraries";
import { PipelineType } from "../../js/src/lib/interfaces/Types";

export interface ExampleRepo {
	description: string;
	id: string;
}

export type TaskDemoEntry = {
	filename: string;
	type: "audio";
} | {
	data: Array<{
		label: string;
		score: number;
	}>;
	type: "chart";
} | {
	filename: string;
	type: "img";
} | {
	content: string;
	label: string;
	type: "text";
} | {
	text: string;
	tokens: Array<{
		end: number;
		start: number;
		type: string;
	}>;
	type: "text-with-tokens";
} ;

export interface TaskDemo {
	inputs: TaskDemoEntry[];
	outputs: TaskDemoEntry[];
}

export interface TaskData {
	datasets: ExampleRepo[];
	demo: TaskDemo;
	id: keyof typeof PipelineType;
	label: string;
	libraries: Array<keyof typeof ModelLibrary>;
	metrics: ExampleRepo[];
	models: ExampleRepo[];
	summary: string;
	widgetModels: string[];
	youtubeId: string;
}
