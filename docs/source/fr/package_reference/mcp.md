# MCP Client

La bibliothèque `huggingface_hub` inclut maintenant un [`MCPClient`], conçu pour donner aux Large Language Models (LLMs) la capacité d'interagir avec des Tools externes via le [Model Context Protocol](https://modelcontextprotocol.io) (MCP). Ce client étend un [`AsyncInferenceClient`] pour intégrer de manière transparente l'utilisation de Tools.

Le [`MCPClient`] se connecte à des serveurs MCP (scripts `stdio` locaux ou services `http`/`sse` distants) qui exposent des tools. Il fournit ces tools à un LLM (via [`AsyncInferenceClient`]). Si le LLM décide d'utiliser un tool, [`MCPClient`] gère la requête d'exécution vers le serveur MCP et relaie la sortie du Tool vers le LLM, souvent en diffusant les résultats en temps réel.

Nous fournissons également une classe [`Agent`] de niveau supérieur. Ce 'Tiny Agent' simplifie la création d'Agents conversationnels en gérant la boucle de chat et l'état, agissant comme un wrapper autour de [`MCPClient`].



## MCP Client

[[autodoc]] MCPClient

## Agent

[[autodoc]] Agent
