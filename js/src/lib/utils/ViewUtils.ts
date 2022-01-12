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
export function escape(html: unknown): string {
	return String(html).replace(/["'&<>]/g, match => ESCAPED[match]);
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
* Converts hex string to rgb array (i.e. [r,g,b])
* original from https://stackoverflow.com/a/39077686/6558628
*/
export function hexToRgb(hex: string): number[]{
	return hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i
			   ,(_, r, g, b) => '#' + r + r + g + g + b + b)
	  .substring(1).match(/.{2}/g)
	  ?.map(x => parseInt(x, 16)) || [0, 0, 0];
}

/*
* For Tailwind:
bg-blue-100 border-blue-100 dark:bg-blue-800 dark:border-blue-800
bg-green-100 border-green-100 dark:bg-green-800 dark:border-green-800
bg-yellow-100 border-yellow-100 dark:bg-yellow-800 dark:border-yellow-800
bg-purple-100 border-purple-100 dark:bg-purple-800 dark:border-purple-800
bg-red-100 border-red-100 dark:bg-red-800 dark:border-red-800
*/
