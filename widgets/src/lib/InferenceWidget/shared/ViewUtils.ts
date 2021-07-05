
const ESCAPED = {
    '"': '&quot;',
    "'": '&#39;',
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;'
};

/**
 *  Returns a function that clamps input value to range [min <= x <= max].
 */
 export function clip(x: number, min: number, max: number): number {
	return Math.max(min, Math.min(x, max));
}

/**
 * HTML escapes the passed string
 */
export function escape(html) {
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
 * Return a unique-ish random id string
 */
export function randomId(prefix = '_'): string {
	// Math.random should be unique because of its seeding algorithm.
	// Convert it to base 36 (numbers + letters), and grab the first 9 characters
	// after the decimal.
	return `${prefix}${Math.random().toString(36).substr(2, 9)}`;
};

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
	if (!x || typeof x !== 'string') {
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
	return arg !== null && typeof arg === 'object' && !Array.isArray(arg);
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
export function scrollToMax(elt: HTMLElement, axis: 'x' | 'y' = 'y') {
	elt.scroll({
		behavior: "smooth",
		left: axis === 'x' ? elt.scrollWidth : undefined,
		top: axis === 'y' ? elt.scrollHeight : undefined,
	});
}
