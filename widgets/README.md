<h1 align="center">🤗-widgets</h1>

<p align="center">
  <a href="https://huggingface-widgets.netlify.app/"><img src="https://img.shields.io/badge/demo_page-Netlify-008080.svg" alt="demo page with Netlify"></a>
  <a href="https://github.com/huggingface/huggingface_hub/actions/workflows/js-widgets-tests.yml"><img src="https://github.com/huggingface/huggingface_hub/actions/workflows/js-widgets-tests.yml/badge.svg" alt="Build Status"></a>
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
npm publish
```

