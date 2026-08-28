from . import constants


class ResolvedRevision(str):
    """A git revision that has already been resolved to a commit hash.

    `ResolvedRevision` is a `str` subclass, so it can be passed to any `huggingface_hub` method taking a `revision`
    argument. Its string value is the revision initially requested by the user (e.g. `"main"`, `"refs/pr/4"`),
    which keeps URLs and error messages readable, while `.resolved` holds the commit hash it points to.

    Instances are built by [`HfApi.resolve_revision`], which also caches the `revision` -> `commit hash` mapping
    in the local cache (`refs/` folder).

    A commit hash only means something for the repo it was resolved against, so an instance also remembers that
    repo. Re-resolving it for another repo is not an error: the revision initially requested is resolved again
    (see [`HfApi.resolve_revision`]).

    Attributes:
        initial (`str` or `None`):
            The revision initially requested by the user. If `None`, the string value defaults to `"main"`.
        resolved (`str`):
            The commit hash that `initial` resolves to.

    Example:
    ```python
    >>> from huggingface_hub import resolve_revision
    >>> revision = resolve_revision("openai-community/gpt2")
    >>> revision
    ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    >>> revision == "main"  # it's a string
    True
    >>> revision.resolved
    '607a30d783dfa663caf39e06633721c8d4cfcd7e'
    ```
    """

    initial: str | None
    resolved: str
    _repo_id: str | None
    _repo_type: str

    def __new__(
        cls,
        resolved: str,
        initial: str | None = None,
        repo_id: str | None = None,
        repo_type: str | None = None,
    ) -> "ResolvedRevision":
        revision = super().__new__(cls, initial if initial is not None else constants.DEFAULT_REVISION)
        revision.initial = initial
        revision.resolved = resolved
        # The repo `resolved` belongs to. `None` means unknown, in which case it is assumed to fit any repo.
        revision._repo_id = repo_id
        revision._repo_type = repo_type or constants.REPO_TYPE_MODEL
        return revision

    def __reduce__(self):
        # without this, pickle/copy rebuild the instance from its string value only, losing the attributes
        return self.__class__, (self.resolved, self.initial, self._repo_id, self._repo_type)

    def __repr__(self) -> str:
        return f"ResolvedRevision(initial={self.initial!r}, resolved={self.resolved!r})"
