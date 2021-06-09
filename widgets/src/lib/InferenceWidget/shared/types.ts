import type { ModelData } from '../../../../lib/FrontData';

export interface WidgetProps {
	apiToken: string;
	apiUrl: string;
	callApiOnMount: boolean;
	model: ModelData;
	noTitle: boolean;
	shouldUpdateUrl: boolean;
}


export type LoadingStatus = "error" | "loaded" | "unknown";
