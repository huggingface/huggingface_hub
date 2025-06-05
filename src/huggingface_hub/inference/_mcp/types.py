from typing import Dict, List, Literal, TypedDict, Union


# Input config
class InputConfig(TypedDict, total=False):
    id: str
    description: str
    type: str
    password: bool


# stdio server config
class StdioServerConfig(TypedDict, total=False):
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: str


class StdioServer(TypedDict):
    type: Literal["stdio"]
    config: StdioServerConfig


# http server config
class HTTPRequestInit(TypedDict, total=False):
    headers: Dict[str, str]


class HTTPServerOptions(TypedDict, total=False):
    requestInit: HTTPRequestInit
    sessionId: str


class HTTPServerConfig(TypedDict, total=False):
    url: str
    options: HTTPServerOptions


class HTTPServer(TypedDict):
    type: Literal["http"]
    config: HTTPServerConfig


# sse server config
class SSEServerOptions(TypedDict, total=False):
    requestInit: HTTPRequestInit


class SSEServerConfig(TypedDict):
    url: str
    options: SSEServerOptions


class SSEServer(TypedDict):
    type: Literal["sse"]
    config: SSEServerConfig


# AgentConfig root object
class AgentConfig(TypedDict):
    model: str
    provider: str
    inputs: List[InputConfig]
    servers: List[Union[StdioServer, HTTPServer, SSEServer]]
