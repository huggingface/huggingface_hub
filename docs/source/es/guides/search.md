<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Buscar en el Hub

En este tutorial aprenderás cómo buscar modelos, datasets (conjuntos de datos) y Spaces (espacios de trabajo) usando `huggingface_hub`.

## ¿Cómo enumerar repositorios?

La biblioteca `huggingface_hub` incluye un cliente HTTP [`HfApi`] para interactuar con el Hub.
Entre otras cosas, permite enumerar modelos, datasets y Spaces almacenados en el Hub:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

El output (salida) de [`list_models`] (enumerar modelos) es un iterador de los modelos almacenados en el Hub.

Del mismo modo, puedes usar [`list_datasets`] (enumerar datasets) para enumerar datasets y [`list_spaces`] (enumerar Spaces) para enumerar Spaces.

## ¿Cómo filtrar repositorios?

Enumerar repositorios está muy bien, pero ahora es probable que quieras filtrar lo que estás buscando.
Las funciones auxiliares de enumeración tienen varios atributos como:
- `filter` (filtro)
- `author` (autor)
- `search` (búsqueda)
- `num_parameters` (número de parámetros)
- ...

Veamos un ejemplo de cómo obtener todos los modelos del Hub que hacen clasificación de imagen, que hayan sido entrenados en el dataset ImageNet, y que se ejecutan con PyTorch.

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

También puedes filtrar modelos por el número de parámetros utilizando la misma sintaxis de rango que la interfaz del Hub:

```py
models = hf_api.list_models(num_parameters="min:6B,max:128B")
```

Mientras filtras, también puedes ordenar los modelos y quedarte solo con los resultados que mejor se adecúen a tu búsqueda. A modo de ejemplo, el siguiente caso obtiene los 5 datasets más descargados del Hub:

```py
>>> list(list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```


Si quieres explorar los filtros disponibles en el Hub, visita las páginas de [modelos](https://huggingface.co/models) y [datasets](https://huggingface.co/datasets) en tu navegador, busca algunos parámetros y observa los valores en la URL.

## ¿Cómo usar la interfaz de línea de comandos (CLI)?

También puedes enumerar y buscar modelos, datasets y Spaces utilizando la interfaz de línea de comandos `hf`:

```bash
# Enumerar modelos
>>> hf models ls --search "llama" --sort downloads --limit 5

# Enumerar datasets
>>> hf datasets ls --author Qwen

# Enumerar Spaces
>>> hf spaces ls --search "3d"

# Obtén información de un modelo específico
>>> hf models info Lightricks/LTX-2
```

Para más detalles, consulta la [guía de la CLI](./cli.md#hf-models).
