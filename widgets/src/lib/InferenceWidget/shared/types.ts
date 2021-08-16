import type { ModelData } from '../../interfaces/Types';

export interface WidgetProps {
	apiToken?: string;
	apiUrl: string;
	callApiOnMount: boolean;
	model: ModelData;
	noTitle: boolean;
	shouldUpdateUrl: boolean;
}


export type LoadingStatus = "error" | "loaded" | "unknown";

export type TableData = Record<string, (string | number)[]>;

export type HighlightCoordinates = Record<string, string>;