# JSONDecodeError was introduced in requests=2.27 released in 2022.
# This allows us to support older requests for users
# More information: https://github.com/psf/requests/pull/5856
try:
    from requests import JSONDecodeError  # noqa
except ImportError:
    try:
        from simplejson import JSONDecodeError  # noqa
    except ImportError:
        from json import JSONDecodeError  # noqa

import yaml


def yaml_dump(data):
    """
    Fix Yaml dump to encode special characters properly.

    Taken from https://stackoverflow.com/a/64566452
           and https://stackoverflow.com/a/4004439

    Example:
    ```py
    >>> yaml.dump({"emoji": "👀", "some unicode": "日本か"})
    'emoji: "\\U0001F440"\nsome unicode: "\\u65E5\\u672C\\u304B"\n'

    >>> yaml_dump({"emoji": "👀", "some unicode": "日本か"})
    'emoji: "👀"\nsome unicode: "日本か"\n'
    ```
    """
    return yaml.dump(data).encode().decode("unicode-escape")
