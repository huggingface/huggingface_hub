import type { ModelData } from '../../../interfaces/Types';

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

type Box = {
	xmin: number;
	ymin: number;
	xmax: number;
	ymax: number;
};

export type DetectedObject = {
	box: Box;
	label: string;
	score: number;
	color?: string;
}
export interface ImageSegment {
	label: string;
	score: number;
	mask: string;
	color?: string;
	imgData?: ImageData;
	bitmap?: ImageBitmap;
};
