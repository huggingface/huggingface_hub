import warnings
from functools import wraps
from inspect import Parameter, signature


def _deprecate_positional_args(func=None, *, version="0.8"):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Args:
        func (``Callable``, `optional`):
            Function to check arguments on.
        version (``Callable``, `optional`, defaults to ``"0.8"``):
            The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []
        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)
            # extra_args > 0
            args_msg = [
                f"{name}='{arg}'" if isinstance(arg, str) else f"{name}={arg}"
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                f"Pass {args_msg} as keyword args. From version "
                f"{version} passing these as positional arguments "
                "will result in an error",
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)
    return _inner_deprecate_positional_args
