# JSONDecodeError was introduced in requests=2.27 released in 2022.
# This allows us to support older requests for users
# More information: https://github.com/psf/requests/pull/5856
try:
    from requests import JSONDecodeError  # type: ignore  # noqa: F401
except ImportError:
    try:
        from simplejson import JSONDecodeError  # type: ignore # noqa: F401
    except ImportError:
        from json import JSONDecodeError  # type: ignore  # noqa: F401

from functools import partial
from typing import Callable

import yaml


# Wrap `yaml.dump` to set `allow_unicode=True` by default.
#
# Example:
# ```py
# >>> yaml.dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "\\U0001F440"\nsome unicode: "\\u65E5\\u672C\\u304B"\n'
#
# >>> yaml_dump({"emoji": "👀", "some unicode": "日本か"})
# 'emoji: "👀"\nsome unicode: "日本か"\n'
# ```
yaml_dump: Callable[..., str] = partial(  # type: ignore
    yaml.dump, stream=None, allow_unicode=True
)
