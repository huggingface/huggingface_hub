import preprocess from "svelte-preprocess";
import adapter from "@sveltejs/adapter-static";

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://github.com/sveltejs/svelte-preprocess
	// for more information about preprocessors
	preprocess: preprocess({
		defaults: { script: "typescript" },
		postcss: true,
	}),

	kit: {
		// hydrate the <div id="svelte"> element in src/app.html
		target: "#svelte",
		adapter: adapter(),
	},
};

export default config;
