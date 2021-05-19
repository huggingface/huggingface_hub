---
title: Model Hub docs
---

<h1 class="no-top-margin">Model Hub documentation</h1>


## Can I write \\( \LaTeX \\) in my model card?

Yes, we use the [KaTeX](https://katex.org/) math typesetting library to render math formulas server-side,
before parsing the markdown.
You have to use the following delimiters:
- `$$ ... $$` for display mode
- `\\` `(` `...` `\\` `)` for inline mode (no space between the slashes and the parenthesis).

Then you'll be able to write:

$$
mse = (\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}
$$

$$ e=mc^2 $$


