<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Inicio rápido

El [Hugging Face Hub](https://huggingface.co/) es el lugar de referencia para compartir
modelos de aprendizaje automático, demos, datasets y métricas. La librería `huggingface_hub`
te ayuda a interactuar con el Hub sin salir de tu entorno de desarrollo. Puedes crear y
gestionar repositorios fácilmente, descargar y subir archivos, y obtener metadatos útiles de
modelos y datasets del Hub.

## Instalación

Para empezar, instala la librería `huggingface_hub`:

```bash
pip install --upgrade huggingface_hub
```

Para más detalles, consulta la guía de [instalación](installation).

> [!TIP]
> `huggingface_hub` también incluye una [CLI `hf`](./guides/cli) que te permite interactuar con el Hub directamente desde la terminal.
> Si usas agentes de IA (Claude Code, Codex, Cursor, ...), instala la Skill para que tu agente pueda usar la CLI:
> ```bash
> # para Codex, Cursor, OpenCode, Pi y otros agentes que cargan skills desde `.agents/skills`
> hf skills add
> # incluye lo anterior + Claude Code
> hf skills add --claude
> ```
> Consulta la guía [Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/agents-cli) para más detalles.

## Descargar archivos

Los repositorios del Hub están versionados con git, y los usuarios pueden descargar un solo
archivo o el repositorio completo. Puedes usar la función [`hf_hub_download`] para descargar
archivos. Esta función descarga y guarda en caché un archivo en tu disco local. La próxima
vez que necesites ese archivo, se cargará desde la caché, así que no necesitas volver a
descargarlo.

Necesitarás el id del repositorio y el nombre del archivo que quieras descargar. Por ejemplo,
para descargar el archivo de configuración del modelo
[Pegasus](https://huggingface.co/google/pegasus-xsum):

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

Para descargar una versión concreta del archivo, usa el parámetro `revision` para indicar el
nombre de la rama, la etiqueta o el hash del commit. Si decides usar el hash del commit, debe
ser el hash completo en lugar del hash corto de 7 caracteres:

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

Para más detalles y opciones, consulta la referencia de la API de [`hf_hub_download`].

<a id="login"></a> <!-- backward compatible anchor -->

## Autenticación

En muchos casos, debes estar autenticado con una cuenta de Hugging Face para interactuar con
el Hub: descargar repositorios privados, subir archivos, crear PRs, ...
[Crea una cuenta](https://huggingface.co/join) si todavía no tienes una.

### Comando de inicio de sesión

La forma más sencilla de autenticarte es con el comando [`login`]:

```bash
hf auth login
```

Si ya has iniciado sesión, el comando volverá de inmediato. Para forzar un nuevo inicio de
sesión, usa `hf auth login --force`. Si no has iniciado sesión, se te pedirá que lo hagas con
tu navegador: abre la URL que se muestra, introduce el código corto, aprueba la solicitud y se
obtendrá un token de acceso que se guardará en tu directorio `HF_HOME` (por defecto
`~/.cache/huggingface/token`). El token caduca al cabo de un tiempo, pero se renueva
automáticamente mientras lo sigas usando. Cualquier script o librería que interactúe con el
Hub usará este token al enviar peticiones. Como alternativa, puedes pegar un [token de acceso
de usuario](https://huggingface.co/docs/hub/security-tokens) generado desde tu [página de
Ajustes](https://huggingface.co/settings/tokens).

> [!TIP]
> Los tokens de acceso de usuario pueden tener permisos de `read` (lectura) o `write` (escritura). Asegúrate de tener un token de tipo `write` si quieres crear o editar un repositorio. En caso contrario, lo mejor es generar un token de tipo `read` para reducir el riesgo si tu token se filtra por accidente.

Como alternativa, puedes iniciar sesión por código usando [`login`] en un notebook o un script:

```py
>>> from huggingface_hub import login
>>> login()
```

Solo puedes tener sesión iniciada en una cuenta a la vez. Iniciar sesión en una cuenta nueva
te cerrará automáticamente la sesión de la anterior. Para saber cuál es tu cuenta activa,
ejecuta el comando `hf auth whoami`.

> [!WARNING]
> Una vez has iniciado sesión, todas las peticiones al Hub (incluso los métodos que no requieren autenticación) usarán tu token de acceso por defecto. Si quieres desactivar el uso implícito de tu token, debes definir la variable de entorno `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` (consulta la [referencia](../package_reference/environment_variables#hfhubdisableimplicittoken)).

### Gestionar varios tokens en local

Puedes guardar varios tokens en tu máquina simplemente iniciando sesión con el comando
[`login`] con cada token. Si necesitas cambiar entre estos tokens en local, puedes usar el
comando [`auth switch`]:

```bash
hf auth switch
```

Este comando te pedirá que selecciones un token por su nombre de una lista de tokens
guardados. Una vez seleccionado, el token elegido pasa a ser el token _activo_ y se usará en
todas las interacciones con el Hub.


Puedes listar todos los tokens de acceso disponibles en tu máquina con `hf auth list`.

### Variable de entorno

La variable de entorno `HF_TOKEN` también puede usarse para autenticarte. Esto resulta
especialmente útil en un Space, donde puedes definir `HF_TOKEN` como un [secret del
Space](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

> [!TIP]
> **NUEVO:** Google Colaboratory te permite definir [claves privadas](https://twitter.com/GoogleColab/status/1719798406195867814) para tus notebooks. Define un secret `HF_TOKEN` para autenticarte automáticamente.

La autenticación mediante una variable de entorno o un secret tiene prioridad sobre el token
guardado en tu máquina.

### Parámetros de los métodos

Por último, también es posible autenticarte pasando tu token a cualquier método que acepte
`token` como parámetro.

```
from huggingface_hub import whoami

user = whoami(token=...)
```

En general esto se desaconseja, salvo en un entorno donde no quieras guardar tu token de forma
permanente o si necesitas manejar varios tokens a la vez.

> [!WARNING]
> Ten cuidado al pasar tokens como parámetro. Siempre es mejor práctica cargar el token desde un almacén seguro en lugar de escribirlo directamente en tu código o notebook. Los tokens escritos en el código presentan un riesgo importante de filtración si compartes tu código por accidente.

## Crear un repositorio

Una vez te has registrado e iniciado sesión, crea un repositorio con la función
[`create_repo`]:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

Si quieres que tu repositorio sea privado:

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

Los repositorios privados no serán visibles para nadie excepto para ti.

> [!TIP]
> Para crear un repositorio o subir contenido al Hub, debes proporcionar un token de acceso de usuario que tenga el permiso `write`. Puedes elegir el permiso al crear el token en tu [página de Ajustes](https://huggingface.co/settings/tokens).

## Subir archivos

Usa la función [`upload_file`] para añadir un archivo a tu repositorio recién creado. Debes
indicar:

1. La ruta del archivo que vas a subir.
2. La ruta del archivo en el repositorio.
3. El id del repositorio donde quieres añadir el archivo.

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

Para subir más de un archivo a la vez, échale un vistazo a la guía de [Subida](./guides/upload),
que te presentará varios métodos para subir archivos (con o sin git).

## Próximos pasos

La librería `huggingface_hub` ofrece una forma sencilla de interactuar con el Hub usando
Python. Para saber más sobre cómo gestionar tus archivos y repositorios en el Hub, te
recomendamos leer nuestras [guías prácticas](./guides/overview) para:

- [Gestionar tu repositorio](./guides/repository).
- [Descargar](./guides/download) archivos del Hub.
- [Subir](./guides/upload) archivos al Hub.
- [Buscar en el Hub](./guides/search) el modelo o dataset que quieras.
- [Ejecutar inferencia](./guides/inference) en varios servicios para modelos alojados en el Hugging Face Hub.
