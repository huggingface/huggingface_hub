import os


def normalize(path: str):
    """Normalizes a path to pass in a URL"""
    return os.path.normpath(path).replace(os.sep, "/").lstrip("/")
