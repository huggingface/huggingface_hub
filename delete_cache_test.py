########################################################
from huggingface_hub.utils import scan_cache_dir
from simple_term_menu import TerminalMenu


hf_cache_info = scan_cache_dir()

to_display = []
commit_hashes = []
for repo in sorted(hf_cache_info.repos, key=lambda r: (r.repo_type, r.repo_id)):
    to_display.append(f"{repo.repo_type}: {repo.repo_id}")
    commit_hashes.append(None)
    for revision in sorted(repo.revisions, key=lambda r: r.commit_hash):
        to_display.append(
            f"  {revision.commit_hash} ({', '.join(revision.refs or 'detached')})"
        )
        commit_hashes.append(revision.commit_hash)


def preview_command(_):
    indices = terminal_menu._selection.selected_menu_indices  # hacky but ok
    selected_hashes = [commit_hashes[item] for item in indices]
    selected_hashes = [item for item in selected_hashes if item is not None]
    if len(selected_hashes) == 0:
        return "No revision selected."
    strategy = hf_cache_info.delete_revisions(*selected_hashes)
    return (
        f"Selected {len(selected_hashes)} revisions to delete. Would save"
        f" {strategy.expected_freed_size_str}."
    )


terminal_menu = TerminalMenu(
    to_display,
    multi_select=True,
    show_multi_select_hint=True,
    preselected_entries=[1, 5, 9],
    title="Choose revisions to delete",
    preview_command=preview_command,
    multi_select_select_on_accept=False,
)
terminal_menu.chosen_menu_indices

selected = terminal_menu.show()
