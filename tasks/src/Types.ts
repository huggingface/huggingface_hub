import type { ModelLibrary } from "../../widgets/src/lib/interfaces/Libraries";
import { PipelineType } from "../../widgets/src/lib/interfaces/Types";

export interface ExampleRepo {
	description: string;
	id: string;
}

export type TaskDemoEntry = {
	content: string;
	label: string;
	type: "text";
} | {
	filename: string;
	type: "img";
} | {
	filename: string;
	type: "audio";
};

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
}
