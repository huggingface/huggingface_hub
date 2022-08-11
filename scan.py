from huggingface_hub.utils.cache_explorer import (
    CachedRepoInfo,
    CachedRevisionInfo,
    scan_cache_dir,
)


hf_cache_info = scan_cache_dir()


def print_revision(revision: CachedRevisionInfo) -> None:
    name = "(detached)" if revision.refs is None else ", ".join(sorted(revision.refs))
    print("        " + revision.commit_hash + ": " + name)
    print(
        "          "
        + revision.size_on_disk_str
        + " ("
        + str(revision.nb_files)
        + " files)"
    )


def print_cached_repo_info(cached_repo: CachedRepoInfo) -> None:
    print(
        "  "
        + cached_repo.repo_id
        + " "
        + cached_repo.size_on_disk_str
        + " ("
        + str(cached_repo.nb_files)
        + " files)"
    )
    print("    " + str(cached_repo.repo_path))
    print("    Revisions:")

    referenced = [
        revision for revision in cached_repo.revisions if revision.refs is not None
    ]
    for revision in referenced:
        print_revision(revision)

    detached = [revision for revision in cached_repo.revisions if revision.refs is None]
    for revision in detached:
        print_revision(revision)


print("Datasets:")
for cached_repo in hf_cache_info.datasets:
    print_cached_repo_info(cached_repo)

print("\nModels:")
for cached_repo in hf_cache_info.models:
    print_cached_repo_info(cached_repo)
