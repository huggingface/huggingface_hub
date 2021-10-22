import type { TableData } from "./types";

const ESCAPED = {
	'"': "&quot;",
	"'": "&#39;",
	"&": "&amp;",
	"<": "&lt;",
	">": "&gt;",
};

/**
 *  Returns a function that clamps input value to range [min <= x <= max].
 */
 export function clamp(x: number, min: number, max: number): number {
	return Math.max(min, Math.min(x, max));
}

/**
 * HTML escapes the passed string
 */
export function escape(html: string) {
	return String(html).replace(/["'&<>]/g, (match) => ESCAPED[match]);
}

/**
 * Returns a promise that will resolve after the specified time
 * @param ms Number of ms to wait
 */
export function delay(ms: number): Promise<void> {
	return new Promise((resolve) => {
		setTimeout(() => resolve(), ms);
	});
}

/**
 * Return a unique-ish random id string
 */
export function randomId(prefix = "_"): string {
	// Math.random should be unique because of its seeding algorithm.
	// Convert it to base 36 (numbers + letters), and grab the first 9 characters
	// after the decimal.
	return `${prefix}${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * "Real" modulo (always >= 0), not remainder.
 */
export function mod(a: number, n: number): number {
	return ((a % n) + n) % n;
}

/**
 * Sum of elements in array
 */
export function sum(arr: number[]): number {
	return arr.reduce((a, b) => a + b, 0);
}

/**
 * Return a random item from an array
 */
export function randomItem<T>(arr: T[]): T {
	return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * Safely parse JSON
 */
export function parseJSON<T>(x: unknown): T | undefined {
	if (!x || typeof x !== "string") {
		return undefined;
	}
	try {
		return JSON.parse(x);
	} catch (e) {
		if (e instanceof SyntaxError) {
			console.error(e.name);
		} else {
			console.error(e.message);
		}
		return undefined;
	}
}

/*
 * Check if a value is a dictionary-like object
 */
export function isObject(arg: any): arg is object {
	return arg !== null && typeof arg === "object" && !Array.isArray(arg);
}

/*
 * Return true if an HTML element is scrolled all the way
 */
export function isFullyScrolled(elt: HTMLElement) {
	return elt.scrollHeight - Math.abs(elt.scrollTop) === elt.clientHeight;
}

/*
 * Smoothly scroll an element all the way
 */
export function scrollToMax(elt: HTMLElement, axis: "x" | "y" = "y") {
	elt.scroll({
		behavior: "smooth",
		left: axis === "x" ? elt.scrollWidth : undefined,
		top: axis === "y" ? elt.scrollHeight : undefined,
	});
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

/*
* For Tailwind:
bg-blue-100 border-blue-100 dark:bg-blue-800 dark:border-blue-800
bg-green-100 border-green-100 dark:bg-green-800 dark:border-green-800
bg-yellow-100 border-yellow-100 dark:bg-yellow-800 dark:border-yellow-800
bg-purple-100 border-purple-100 dark:bg-purple-800 dark:border-purple-800
bg-red-100 border-red-100 dark:bg-red-800 dark:border-red-800
*/
