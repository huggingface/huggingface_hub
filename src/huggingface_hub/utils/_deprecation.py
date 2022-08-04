import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Callable, Set


def _deprecate_positional_args(func: Callable, *, version: str) -> Callable:
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Args:
        func (``Callable``):
            Function to check arguments on.
        version (``str``):
            The version when positional arguments will result in error.
    """
    sig = signature(func)
    kwonly_args = []
    all_args = []
    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(func)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args <= 0:
            return func(*args, **kwargs)
        # extra_args > 0
        args_msg = [
            f"{name}='{arg}'" if isinstance(arg, str) else f"{name}={arg}"
            for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
        ]
        args_msg = ", ".join(args_msg)
        warnings.warn(
            f"Pass {args_msg} as keyword args. From version {version} passing these as "
            "positional arguments will result in an error,",
            FutureWarning,
        )
        kwargs.update(zip(sig.parameters, args))
        return func(**kwargs)

    return inner_f


def _deprecate_arguments(
    func: Callable, *, version: str, deprecated_args: Set[str]
) -> Callable:
    """Decorator to issue warnings when using deprecated arguments.

    Args:
        func (``Callable``):
            Function to check arguments on.
        version (``str``):
            The version when deprecated arguments will result in error.
        deprecated_args (``List[str]``):
            List of the arguments to be deprecated.
    """
    sig = signature(func)

    @wraps(func)
    def inner_f(*args, **kwargs):
        used_deprecated_args = []

        for _, parameter in zip(args, sig.parameters.values()):
            if parameter.name in deprecated_args:
                used_deprecated_args.append(parameter.name)

        for kwarg_name in kwargs:
            if kwarg_name in deprecated_args:
                used_deprecated_args.append(kwarg_name)

        if len(used_deprecated_args) > 0:
            warnings.warn(
                f"Deprecated argument(s) used in '{func.__name__}':"
                f" {', '.join(used_deprecated_args)}. Will not be supported from"
                f" version '{version}'.",
                FutureWarning,
            )

        return func(*args, **kwargs)

    return inner_f
