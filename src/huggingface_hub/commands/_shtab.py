"""Fake shtab."""
from argparse import Action, ArgumentParser


_WARNING_MESSAGE = (
    "You first need to install `shtab` to use autocompletion. Please run `pip install"
    " 'huggingface_hub[completion]'`"
)


class PrintCompletionAction(Action):
    """Fake `PrintCompletionAction` when shtab is not installed.

    See <https://github.com/iterative/shtab/blob/95b0e3092cd4dcf1ac2871d44cebda01a89992df/shtab/__init__.py#L786-L789>
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """Warn when shtab is not installed.

        :param parser:
        :param namespace:
        :param values:
        :param option_string:
        """
        print(_WARNING_MESSAGE)
        parser.exit(0)


def add_argument_to(parser: ArgumentParser, *args, **kwargs):
    """Add completion argument to parser.

    :param parser:
    :type parser: ArgumentParser
    :param args:
    :param kwargs:
    """
    Action.complete = None  # type: ignore
    parser.add_argument(
        "--print-completion",
        choices=["bash", "zsh", "tcsh"],
        action=PrintCompletionAction,
        help=_WARNING_MESSAGE,
    )
    return parser
