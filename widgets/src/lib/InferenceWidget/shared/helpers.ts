import type { ModelData } from '$lib/interfaces/Types';
import { randomItem, parseJSON, } from './ViewUtils';
import type { LoadingStatus } from './types';

export function getSearchParams(keys: string[]): string[] {
	const searchParams = new URL(window.location.href).searchParams;
	return keys.map((key) => {
		const value = searchParams.get(key);
		return value ? value : '';
	});
}

export function getDemoInputs(model: ModelData, keys: (number | string)[]): any[] {
	const widgetData = Array.isArray(model.widgetData) ? model.widgetData : [];
	const randomEntry = randomItem(widgetData) ?? {};
	return keys.map((key) => {
		const value = (randomEntry[key])
			? randomEntry[key]
			: null;
		return value ? randomEntry[key] : null;
	});
}

// Update current url search params, keeping existing keys intact.
export function updateUrl(obj: Record<string, string>) {
	if (!window) {
		return;
	}

	const sp = new URL(window.location.href).searchParams;
	for (const [k, v] of Object.entries(obj)) {
		if (v === undefined) {
			sp.delete(k);
		} else {
			sp.set(k, v);
		}
	}
	const path = `${window.location.pathname}?${sp.toString()}`;
	window.history.replaceState(null, "", path);
}

async function callApi(
	url: string, 
	modelId: string, 
	requestBody: Record<string, any>, 
	apiToken = '',
	waitForModel = false, // If true, the server will only respond once the model has been loaded on the inference API,
	useCache = true,
): Promise<Response> {	
	const contentType = 'file' in requestBody && 'type' in requestBody['file']
		? requestBody['file']['type']  
		: 'application/json';
	
	const headers = new Headers();
	headers.set('Content-Type', contentType);
	if (apiToken) {
		headers.set("Authorization", `Bearer ${apiToken}`);
	}
	if (waitForModel) {
		headers.set("X-Wait-For-Model", "true");
	}
	if (useCache === false) {
		headers.set('X-Use-Cache', "false");
	}
	
	const body: File | string = 'file' in requestBody
		? requestBody.file
		: JSON.stringify(requestBody);
	
	return await fetch(
		`${url}/models/${modelId}`,
		{
			method: "POST",
			body,
			headers,
		}
	);
}

export async function getResponse<T>(
	url: string, 
	modelId: string, 
	requestBody: Record<string, any>, 
	apiToken = '',
	outputParsingFn: (x: unknown) =>  T,
	waitForModel = false, // If true, the server will only respond once the model has been loaded on the inference API,
	useCache = true,
): Promise<{
	computeTime: string,
	output: T,
	outputJson: string,
	response: Response,
	status: 'success'
} | {
	error: string,
	estimatedTime: number,
	status: 'loading-model'
} | {
	error: string,
	status: 'error'
}>  {
	const response = await callApi(
		url,
		modelId,
		requestBody,
		apiToken,
		waitForModel,
		useCache,
	);

	if (response.ok) {
		// Success
		const computeTime = response.headers.has("x-compute-time")
			? `${response.headers.get("x-compute-time")} s`
			: `cached`;
		const isMediaContent = (response.headers.get('content-type')?.search(/^(?:audio|image)/i) ?? -1) !== -1;
		
		const body = !isMediaContent 
			? await response.json()
			: await response.blob();
		const output = outputParsingFn(body);
		const outputJson = !isMediaContent ? JSON.stringify(body, null, 2) : '';
		
		return { computeTime, output, outputJson, response, status: 'success' }
	} else {
		// Error
		const bodyText = await response.text();
		const body = parseJSON<Record<string, any>>(bodyText) ?? {};

		if (
			body["error"] &&
			response.status === 503 &&
			body["estimated_time"] != null // != null -> check for null AND undefined
		) {
			// Model needs loading
			return { error: body["error"], estimatedTime: body["estimated_time"], status: 'loading-model' };
		} else {
			// Other errors
			return { error: body["error"] ?? body["traceback"] ?? body, status: 'error' };
		}
	}
}


export async function getModelStatus(url: string, modelId: string): Promise<LoadingStatus> {
	const response = await fetch(`${url}/status/${modelId}`);
	const output = await response.json();
	if (response.ok && typeof output === 'object' && output.loaded !== undefined) {
		return output.loaded ? 'loaded' : 'unknown';
	} else {
		console.warn(response.status, output.error);
		return 'error';
	}
}
