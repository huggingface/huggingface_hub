<h1 align="center">huggingface-widgets</h1>

<p align="center">
  <a href="https://huggingface-widgets.netlify.app/"><img src="https://img.shields.io/badge/demo_page-Netlify-008080.svg" alt="demo page with Netlify"></a>
  <a href="https://github.com/huggingface/huggingface_hub/actions/workflows/js-widgets-tests.yml?query=branch%3Amain"><img src="https://github.com/huggingface/huggingface_hub/actions/workflows/js-widgets-tests.yml/badge.svg?query=branch%3Amain" alt="Build Status"></a>
  <a href="https://github.com/prettier/prettier"><img src="https://img.shields.io/badge/styled_with-prettier-ff69b4.svg" alt="styled with prettier"></a>
  <a href="https://kit.svelte.dev/"><img src="https://img.shields.io/badge/made_with-SvelteKit-ff3e00.svg" alt="made with SvelteKit"></a>
  <a href="https://github.com/huggingface/huggingface_hub/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-informational" alt="License"></a>
</p>

# 

Open-source version of the inference widgets from huggingface.co. Widgets allow anyone to do inference directly on the browser! For more information about widgets, please go to our [documentation](https://huggingface.co/docs/hub/main#whats-a-widget).

## How to develop

Once you've created a project and installed dependencies with `npm install` (or `pnpm install` or `yarn`), start a development server:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```

## Build for Netlify

```bash
npm run build
```
## Publish package

```bash
npm run publish
```

## Contribution guideline
1. Create a new branch to which you will be pushing your updates
2. Use descriptive name for your branch (e.g. `widget_object_detection`)
3. Create your widget in `src/lib/InferenceWidget/widgets/[MyNewWidget].svelte`
4. Try to use as many components as possible from [`shared`](https://github.com/huggingface/huggingface_hub/tree/main/widgets/src/lib/InferenceWidget/shared)
5. For the API contract, check out [Accelerated Inference API doc](https://api-inference.huggingface.co/docs/python/html/detailed_parameters.html) & [Integrating your library to the Hub](https://huggingface.co/docs/hub/adding-a-library)
6. For your widget, make sure to implement [getOutput](https://github.com/huggingface/huggingface_hub/blob/main/widgets/src/lib/InferenceWidget/widgets/ImageClassificationWidget/ImageClassificationWidget.svelte#L28), [isValidOutput](https://github.com/huggingface/huggingface_hub/blob/main/widgets/src/lib/InferenceWidget/widgets/ImageClassificationWidget/ImageClassificationWidget.svelte#L69), [parseOutput](https://github.com/huggingface/huggingface_hub/blob/main/widgets/src/lib/InferenceWidget/widgets/ImageClassificationWidget/ImageClassificationWidget.svelte#L78) ([raise TypeError](https://github.com/huggingface/huggingface_hub/blob/main/widgets/src/lib/InferenceWidget/widgets/ImageClassificationWidget/ImageClassificationWidget.svelte#L82) on invalid output), and all other necessary functions for your widget to work
7. Submit a PR when done (or draft PR if you are still working on it & would like to collaborate)
8. See [previous widget PRs](https://github.com/huggingface/huggingface_hub/pulls?q=is%3Apr+label%3Awidgets+) 