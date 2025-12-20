import typer
from click import Context
from typer.core import TyperGroup

from huggingface_hub.utils import is_transformers_available


class LazyTransformersGroup(TyperGroup):
    """Lazy loading group for transformers CLI.

    TODO: if we add more lazy loaded CLIs we should generalize this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._real_group = None

    def _load_real_group(self):
        if self._real_group is None:
            if not is_transformers_available():
                raise ImportError("Transformers is not installed. Please install it with: pip install transformers")
            from transformers.cli.transformers import app as real_transformers_cli
            from typer.main import get_group

            self._real_group = get_group(real_transformers_cli)
        return self._real_group

    def list_commands(self, ctx: Context):
        try:
            real_group = self._load_real_group()
            return real_group.list_commands(ctx)
        except ImportError:
            return []

    def get_command(self, ctx: Context, cmd_name: str):
        try:
            real_group = self._load_real_group()
            return real_group.get_command(ctx, cmd_name)
        except ImportError as e:
            ctx.fail(str(e))


lazy_transformers = typer.Typer(
    cls=LazyTransformersGroup, help="Alias for Transformers CLI. Only available if `transformers` is installed."
)
