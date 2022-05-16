"""
lazy_loader
===========

Makes it easy to load subpackages and functions on demand.
"""
import importlib
import importlib.util
import inspect
import os
import sys
import types


__all__ = ["attach", "load"]


def attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they would
    with normal imports, except that the import occurs upon first use.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': ['someattr']}
      )

    This functionality requires Python 3.7 or higher.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    submodules : set
        List of submodules to attach.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        elif name in attr_to_modules:
            submod = importlib.import_module(f"{package_name}.{attr_to_modules[name]}")
            return getattr(submod, name)
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


class DelayedImportErrorModule(types.ModuleType):
    def __init__(self, frame_data, *args, **kwargs):
        self.__frame_data = frame_data
        super().__init__(*args, **kwargs)

    def __getattr__(self, x):
        if x in ("__class__", "__file__", "__frame_data"):
            super().__getattr__(x)
        else:
            fd = self.__frame_data
            raise ModuleNotFoundError(
                f"No module named '{fd['spec']}'\n\nThis error is lazily reported,"
                f' having originally occured in\n  File {fd["filename"]}, line'
                f' {fd["lineno"]}, in {fd["function"]}\n\n---->'
                f' {"".join(fd["code_context"]).strip()}'
            )


def load(fullname, error_on_import=False):
    """Return a lazily imported proxy for a module.

    We often see the following pattern::

      def myfunc():
          from numpy import linalg as la
          la.norm(...)
          ....

    This is to prevent a module, in this case `numpy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      la = lazy.load("numpy.linalg")

      def myfunc():
          la.norm(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of the library.

    Parameters
    ----------
    fullname : str
        The full name of the module or submodule to import.  For example::

          sp = lazy.load('scipy')  # import scipy as sp
          spla = lazy.load('scipy.linalg')  # import scipy.linalg as spla
    error_on_import : bool
        Whether to postpone raising import errors until the module is accessed.
        If set to `True`, import errors are raised as soon as `load` is called.

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except KeyError:
        pass

    spec = importlib.util.find_spec(fullname)
    if spec is None:
        if error_on_import:
            raise ModuleNotFoundError(f"No module named '{fullname}'")
        else:
            try:
                parent = inspect.stack()[1]
                frame_data = {
                    "spec": fullname,
                    "filename": parent.filename,
                    "lineno": parent.lineno,
                    "function": parent.function,
                    "code_context": parent.code_context,
                }
                return DelayedImportErrorModule(frame_data, "DelayedImportErrorModule")
            finally:
                del parent

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module

    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)

    return module
