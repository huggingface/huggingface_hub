import { writable } from "svelte/store";

export let canvas = writable<HTMLCanvasElement>(null);
export let img = writable<HTMLImageElement>(null);
export let width = writable(0);
export let heigth = writable(0);