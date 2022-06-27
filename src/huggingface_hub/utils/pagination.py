from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Pagination(Generic[T]):
    """Utility data structure to manage pagination on the Hub APIs

    Attributes:
        page (`list`):
            The page of resources retrieved from the Hub API.
        total (`int`):
            The total number of resources available.
        page_num (`int`):
            The page index.
        next_page (`function`, *optional*):
            A callable returning the next page of resources, or `None` if this is
            the last page.

    Examples:
        Example usage:
        ```python
        >>> paginated = get_pagination()
        # let's say paginated is an instance of `Pagination[str]`
        >>> paginated.page
        # ["hello", "greetings"]
        >>> next_page = paginated.next_page()
        # next_page is also a instance of `Paginated[int]`
        ```
    """

    page: List[T]
    total: int
    page_num: int
    next_page: Optional[Callable[[], "Pagination[T]"]]

    @property
    def has_next(self) -> bool:
        """Whether or not there is a next page to  this pagination"""
        return self.next_page is not None
