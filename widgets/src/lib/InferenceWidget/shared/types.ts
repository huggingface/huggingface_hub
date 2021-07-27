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
