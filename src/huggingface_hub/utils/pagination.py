from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Pagination(Generic[T]):
    """Utility to manage pagination on the Hub APIs

    Attributes:
        value (`list`):
            The list of resources retrieved from the Hub API.
        total (`int`):
            The total number of resources available.
        page_num (`int`):
            The page index.
        next_page (`function`):
            A callable returning the next page of resources.
    """

    value: List[T]
    total: int
    page_num: int
    next_page: Optional[Callable[[], "Pagination[T]"]]
