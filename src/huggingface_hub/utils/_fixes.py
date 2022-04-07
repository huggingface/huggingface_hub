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
