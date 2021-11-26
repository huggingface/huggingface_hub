import { PipelineType } from "../widgets/src/lib/interfaces/Types";
import type { TaskData } from "./Types";
import questionAnswering from "./question-answering/data";

export const TASKS_DATA: Partial<Record<keyof typeof PipelineType, TaskData>> = {
	"question-answering": questionAnswering,
} as const;
