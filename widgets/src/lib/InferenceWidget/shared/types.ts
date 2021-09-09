import type { ModelData } from '$lib/interfaces/Types';

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

export type Box = {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
};