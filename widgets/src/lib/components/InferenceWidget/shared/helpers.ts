import type { ModelData } from '../../../interfaces/Types';
import { randomItem, parseJSON } from '../../../utils/ViewUtils';
import type { LoadingStatus, TableData } from './types';

export function getSearchParams(keys: string[]): string[] {
	const searchParams = new URL(window.location.href).searchParams;
	return keys.map((key) => {
		const value = searchParams.get(key);
		return value ? value : '';
	});
}

export function getDemoInputs(model: ModelData, keys: (number | string)[]): any[] {
	const widgetData = Array.isArray(model.widgetData) ? model.widgetData : [];
	const randomEntry = (randomItem(widgetData) ?? {}) as any;
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

// Run through our own proxy to bypass CORS:
function proxify(url: string): string {
	return url.startsWith(`http://localhost`)
		|| new URL(url).host === window.location.host
		? url
		: `https://widgets-cors-proxy.huggingface.co/proxy?url=${url}`;
}

// Get BLOB from a given URL after proxifying the URL
export async function getBlobFromUrl(url: string): Promise<Blob>{
	const proxiedUrl = proxify(url);
	const res = await fetch(proxiedUrl);
	const blob = await res.blob();
	return blob;
}

async function callApi(
	url: string, 
	repoId: string, 
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
		`${url}/models/${repoId}`,
		{
			method: "POST",
			body,
			headers,
		}
	);
}

export async function getResponse<T>(
	url: string, 
	repoId: string, 
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
		repoId,
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

		try{
			const output = outputParsingFn(body);
			const outputJson = !isMediaContent ? JSON.stringify(body, null, 2) : '';
			return { computeTime, output, outputJson, response, status: 'success' }
		}catch(e){
			// Invalid output
			const error = `API Implementation Error: ${e.message}`;
			return { error, status: 'error' }
		}
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
			const { status, statusText } = response;
			return { error: body["error"] ?? body["traceback"] ?? `${status} ${statusText}`, status: 'error' };
		}
	}
}


export async function getModelStatus(url: string, repoId: string): Promise<LoadingStatus> {
	const response = await fetch(`${url}/status/${repoId}`);
	const output = await response.json();
	if (response.ok && typeof output === 'object' && output.loaded !== undefined) {
		return output.loaded ? 'loaded' : 'unknown';
	} else {
		console.warn(response.status, output.error);
		return 'error';
	}
}

// Extend Inference API requestBody with user supplied Inference API parameters
export function addInferenceParameters(requestBody: Record<string, any>, model: ModelData) {
	const inference = model?.cardData?.inference;
	if (typeof inference === "object") {
		const inferenceParameters = inference?.parameters;
		if (inferenceParameters) {
			if (requestBody.parameters) {
				requestBody.parameters = { ...requestBody.parameters, ...inferenceParameters };
			} else {
				requestBody.parameters = inferenceParameters;
			}
		}
	}
}

/*
* Converts table from [[Header0, Header1, Header2], [Column0Val0, Column1Val0, Column2Val0], ...]
* to {Header0: [ColumnVal0, ...], Header1: [Column1Val0, ...], Header2: [Column2Val0, ...]}
*/
export function convertTableToData(table: (string | number)[][]): TableData {
	return Object.fromEntries(
		table[0].map((cell, x) => {
			return [
				cell,
				table
					.slice(1)
					.flat()
					.filter((_, i) => i % table[0].length === x)
					.map((x) => String(x)), // some models can only handle strings (no numbers)
			];
		})
	);
}

/*
* Converts data from {Header0: [ColumnVal0, ...], Header1: [Column1Val0, ...], Header2: [Column2Val0, ...]}
* to [[Header0, Header1, Header2], [Column0Val0, Column1Val0, Column2Val0], ...]
*/
export function convertDataToTable(data: TableData): (string | number)[][] {
	const dataArray = Object.entries(data); // [header, cell[]][]
	const nbCols = dataArray.length;
	const nbRows = (dataArray[0]?.[1]?.length ?? 0) + 1;
	return Array(nbRows)
		.fill("")
		.map((_, y) =>
			Array(nbCols)
				.fill("")
				.map((_, x) => (y === 0 ? dataArray[x][0] : dataArray[x][1][y - 1]))
		);
}
