<!--⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Paradigma Git frente a HTTP

La biblioteca `huggingface_hub` es una biblioteca para interactuar con el Hub de Hugging Face, que es una colección de repositorios basados en Git (modelos, datasets o Spaces). Existen dos maneras principales de acceder al Hub a través de `huggingface_hub`.

El primer enfoque, el denominado "basado en Git", consiste en utilizar comandos estándar de `git` directamente en una terminal. Este método te permite clonar repositorios, crear *commits* y enviar cambios manualmente. La segunda opción, llamada "basada en HTTP", implica realizar solicitudes HTTP utilizando el cliente [`HfApi`]. Vamos a examinar los pros y los contras de cada enfoque.

## Git: el enfoque histórico basado en la CLI

Al principio, la mayoría de los usuarios interactuaban con el Hub de Hugging Face utilizando comandos básicos de `git` como `git clone`, `git add`, `git commit`, `git push`, `git tag` o `git checkout`.

Este enfoque te permite trabajar con una copia local completa del repositorio en tu equipo, al igual que en el desarrollo de software tradicional. Esto puede ser una ventaja cuando necesitas acceso sin conexión a internet o cuando quieres trabajar con el historial completo del repositorio. No obstante, este planteamiento también tiene sus desventajas: eres responsable de mantener el repositorio actualizado localmente, gestionar las credenciales y administrar archivos de gran tamaño (a través de `git-lfs`), lo que puede resultar pesado al trabajar con grandes modelos de aprendizaje automático o datasets.

En muchos flujos de trabajo de aprendizaje automático, tal vez solo necesites descargar unos pocos archivos para inferencia o para convertir los pesos sin necesidad de clonar el repositorio completo. En casos así, usar `git` puede ser excesivo e introducir una complejidad innecesaria.

## HfApi: un cliente HTTP flexible y conveniente

La clase [`HfApi`] fue desarrollada para ofrecer una alternativa al uso de repositorios locales de Git, los cuales pueden resultar pesados de mantener, especialmente al trabajar con modelos grandes o datasets. La clase [`HfApi`] ofrece la misma funcionalidad que los flujos de trabajo basados en Git —como descargar y subir archivos, o crear ramas y etiquetas— pero sin la necesidad de una carpeta local que deba mantenerse sincronizada.

Además de las funcionalidades que ya ofrece `git`, la clase [`HfApi`] incluye características adicionales, como la capacidad de gestionar repositorios, descargar archivos utilizando la memoria caché para un reuso eficiente, buscar repositorios y metadatos en el Hub, acceder a funciones de la comunidad como debates, solicitudes de extracción (PRs) y comentarios, y configurar el hardware y las variables secretas de los Spaces.

## ¿Qué debería usar? ¿Y cuándo?

En general, **el enfoque basado en HTTP es el recomendado para usar** `huggingface_hub` en todos los casos. [`HfApi`] te permite obtener y enviar cambios, trabajar con solicitudes de extracción (PRs), etiquetas y ramas, interactuar con discusiones y mucho más.

Sin embargo, no todos los comandos de Git están disponibles en [`HfApi`]. Puede que algunos nunca lleguen a implementarse, pero siempre intentamos mejorar y recortar esa distancia. Si ves que tu caso de uso no está cubierto, ¡por favor, abre una [incidencia (*issue*) en GitHub](https://github.com/huggingface/huggingface_hub)! Agradecemos todos los comentarios para ayudar a construir el ecosistema de HF con y para nuestros usuarios.

Esta preferencia por [`HfApi`] (basado en HTTP) frente a los comandos directos de `git` no significa que el control de versiones con Git vaya a desaparecer del Hub de Hugging Face a corto plazo. Siempre será posible utilizar `git` localmente en aquellos flujos de trabajo en los que tenga sentido.