from typing import Dict, List, Literal, TypedDict, Union

from typing_extensions import NotRequired


class InputConfig(TypedDict, total=False):
    id: str
    description: str
    type: str
    password: bool


class ToolsConfig(TypedDict, total=False):
    include: NotRequired[List[str]]
    exclude: NotRequired[List[str]]


class StdioServerConfig(TypedDict):
    type: Literal["stdio"]
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: str
    tools: NotRequired[ToolsConfig]


class HTTPServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: Dict[str, str]
    tools: NotRequired[ToolsConfig]


class SSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: Dict[str, str]
    tools: NotRequired[ToolsConfig]


ServerConfig = Union[StdioServerConfig, HTTPServerConfig, SSEServerConfig]


# AgentConfig root object
class AgentConfig(TypedDict):
    model: str
    provider: str
    apiKey: NotRequired[str]
    inputs: List[InputConfig]
    servers: List[ServerConfig]
