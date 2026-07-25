<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Instalación

Antes de empezar, tendrás que preparar tu entorno instalando los paquetes adecuados.

`huggingface_hub` se prueba en **Python 3.10+**.

## Instalar con pip

Es muy recomendable instalar `huggingface_hub` en un [entorno virtual](https://docs.python.org/3/library/venv.html).
Si no estás familiarizado con los entornos virtuales de Python, échale un vistazo a esta [guía](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
Un entorno virtual facilita gestionar distintos proyectos y evita problemas de compatibilidad entre dependencias.

Empieza creando un entorno virtual en el directorio de tu proyecto:

```bash
python -m venv .venv
```

Activa el entorno virtual. En Linux y macOS:

```bash
source .venv/bin/activate
```

Activa el entorno virtual en Windows:

```bash
.venv/Scripts/activate
```

Ahora ya puedes instalar `huggingface_hub` [desde el registro de PyPi](https://pypi.org/project/huggingface-hub/):

```bash
pip install --upgrade huggingface_hub
```

Una vez hecho, [comprueba que la instalación](#comprobar-la-instalación) funciona correctamente.

### Instalar dependencias opcionales

Algunas dependencias de `huggingface_hub` son [opcionales](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies) porque no son necesarias para ejecutar las funcionalidades principales de `huggingface_hub`. Sin embargo, algunas funcionalidades de `huggingface_hub` pueden no estar disponibles si no se instalan las dependencias opcionales.

Puedes instalar las dependencias opcionales con `pip`:
```bash
# Instala las dependencias para las funcionalidades específicas de torch y de MCP.
pip install 'huggingface_hub[mcp,torch]'
```

Esta es la lista de dependencias opcionales de `huggingface_hub`:
- `fastai`, `torch`: dependencias para ejecutar funcionalidades específicas de cada framework.
- `dev`: dependencias para contribuir a la librería. Incluye `testing` (para ejecutar los tests), `typing` (para ejecutar el comprobador de tipos) y `quality` (para ejecutar los linters).



### Instalar desde el código fuente

En algunos casos resulta interesante instalar `huggingface_hub` directamente desde el código fuente.
Esto te permite usar la versión `main` más reciente en lugar de la última versión estable.
La versión `main` es útil para estar al día de los últimos desarrollos, por ejemplo
si un error se ha corregido desde la última versión oficial pero todavía no se ha publicado una nueva versión.

Sin embargo, esto significa que la versión `main` no siempre es estable. Nos esforzamos por mantener la
versión `main` operativa, y la mayoría de los problemas suelen resolverse
en unas pocas horas o en un día. Si te encuentras con un problema, abre una Issue para que podamos
corregirlo aún antes.

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

Al instalar desde el código fuente, también puedes indicar una rama concreta. Esto es útil si
quieres probar una nueva funcionalidad o una corrección que aún no se ha fusionado:

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

Una vez hecho, [comprueba que la instalación](#comprobar-la-instalación) funciona correctamente.

### Instalación editable

Instalar desde el código fuente te permite configurar una [instalación editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).
Es una instalación más avanzada, pensada para cuando vas a contribuir a `huggingface_hub`
y necesitas probar cambios en el código. Necesitas clonar una copia local de `huggingface_hub`
en tu máquina.

```bash
# Primero, clona el repositorio en local
git clone https://github.com/huggingface/huggingface_hub.git

# Después, instala con el flag -e
cd huggingface_hub
pip install -e .
```

Estos comandos enlazan la carpeta a la que clonaste el repositorio con las rutas de tus librerías de Python.
Ahora Python buscará dentro de la carpeta que clonaste además de en las rutas habituales de librerías.
Por ejemplo, si tus paquetes de Python se instalan normalmente en `./.venv/lib/python3.13/site-packages/`,
Python también buscará en la carpeta a la que clonaste `./huggingface_hub/`.

## Instalar la CLI de Hugging Face

Usa nuestros instaladores de una línea para configurar la CLI `hf` sin tocar tu entorno de Python:

En macOS y Linux:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

En Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Para actualizar una instalación existente, ejecuta `hf update`: detecta cómo se instaló `hf` (instalador independiente, Homebrew o pip) y ejecuta el comando correspondiente.

## Instalar con conda

Si te resulta más familiar, puedes instalar `huggingface_hub` usando el [canal conda-forge](https://anaconda.org/conda-forge/huggingface_hub):


```bash
conda install -c conda-forge huggingface_hub
```

Una vez hecho, [comprueba que la instalación](#comprobar-la-instalación) funciona correctamente.

## Comprobar la instalación

Una vez instalado, comprueba que `huggingface_hub` funciona correctamente ejecutando el siguiente comando:

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```

Este comando obtendrá del Hub información sobre el modelo [gpt2](https://huggingface.co/gpt2).
La salida debería tener este aspecto:

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Limitaciones en Windows

Con nuestro objetivo de democratizar el buen ML en todas partes, construimos `huggingface_hub` para que sea una
librería multiplataforma y, en particular, para que funcione correctamente tanto en sistemas basados en Unix como en
Windows. Sin embargo, hay algunos casos en los que `huggingface_hub` tiene ciertas limitaciones al
ejecutarse en Windows. Esta es una lista exhaustiva de los problemas conocidos. Háznoslo saber si te
encuentras con algún problema no documentado abriendo [una issue en Github](https://github.com/huggingface/huggingface_hub/issues/new/choose).

- El sistema de caché de `huggingface_hub` se basa en enlaces simbólicos para cachear de forma eficiente los archivos descargados
del Hub. En Windows, debes activar el modo desarrollador o ejecutar tu script como administrador para
habilitar los enlaces simbólicos. Si no están activados, el sistema de caché sigue funcionando, pero de forma no optimizada.
Lee la sección [limitaciones de la caché](./guides/manage-cache#limitations) para más detalles.
- Las rutas de archivo en el Hub pueden tener caracteres especiales (por ejemplo, `"path/to?/my/file"`). Windows es
más restrictivo con los [caracteres especiales](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names),
lo que hace imposible descargar esos archivos en Windows. Por suerte es un caso poco frecuente.
Ponte en contacto con el propietario del repositorio si crees que es un error, o con nosotros para encontrar
una solución.


## Próximos pasos

Una vez `huggingface_hub` está correctamente instalado en tu máquina, quizá quieras
[configurar las variables de entorno](package_reference/environment_variables) o [consultar alguna de nuestras guías](guides/overview) para empezar.
