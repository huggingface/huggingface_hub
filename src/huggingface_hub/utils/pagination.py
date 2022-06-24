from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Pagination(Generic[T]):
    value: List[T]
    total: int
    page_num: int
    next_page: Optional[Callable[[], "Pagination[T]"]]
