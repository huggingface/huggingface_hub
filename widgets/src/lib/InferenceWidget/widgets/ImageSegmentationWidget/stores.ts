import { writable } from "svelte/store";

export let highlightIndex = writable(-1);
export let updateCounter = writable(1);