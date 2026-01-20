# Comprehensive HuggingFace Hub CLI Assessment

## Executive Summary

The `hf` CLI is already a powerful and well-structured tool that covers many core workflows. However, comparing the CLI with the extensive API (`HfApi`) reveals several gaps in functionality, inconsistencies, and opportunities for UX improvements—especially for AI agents that need machine-readable output and predictable behavior.

---

## 1. Current CLI Structure Overview

### Top-Level Commands
- `download` - Download files from the Hub
- `upload` - Upload file/folder to the Hub
- `upload-large-folder` - Resumable large folder uploads
- `env` - Print environment info
- `version` - Print version

### Command Groups
| Group | Commands |
|-------|----------|
| `auth` | `login`, `logout`, `switch`, `list`, `whoami` |
| `cache` | `ls`, `rm`, `prune`, `verify` |
| `datasets` | `ls`, `info` |
| `models` | `ls`, `info` |
| `spaces` | `ls`, `info` |
| `repo` | `create`, `delete`, `move`, `settings`, `tag/*`, `branch/*` |
| `repo-files` | `delete` |
| `jobs` | `run`, `logs`, `stats`, `ps`, `hardware`, `inspect`, `cancel`, `uv/*`, `scheduled/*` |
| `endpoints` | `ls`, `deploy`, `describe`, `update`, `delete`, `pause`, `resume`, `scale-to-zero`, `catalog/*` |

---

## 2. Discrepancies: API Methods Missing from CLI

### High Priority (Frequently Needed)

#### 2.1 Discussion & Pull Request Management
The API has extensive discussion/PR support that's completely missing from CLI:
- `get_repo_discussions` - List discussions/PRs
- `get_discussion_details` - Get details of a discussion/PR
- `create_discussion` - Create a new discussion
- `create_pull_request` - Create a new PR
- `comment_discussion` - Add comment to discussion
- `merge_pull_request` - Merge a PR
- `change_discussion_status` - Open/close discussion
- `rename_discussion` - Rename a discussion

**Suggested CLI group: `hf discussions` or `hf pr`**

#### 2.2 Repository Inspection Commands
- `list_repo_tree` - List files in a repo (with tree structure) 
- `list_repo_commits` - List commit history
- `list_repo_refs` - List branches and tags (currently only accessible via `tag list`)
- `file_exists` / `repo_exists` / `revision_exists` - Quick existence checks

**Suggested commands: `hf repo tree`, `hf repo commits`, `hf repo refs`, `hf repo exists`**

#### 2.3 Collection Management
Collections are a key Hub feature with no CLI support:
- `list_collections` - List collections
- `get_collection` - Get collection details
- `create_collection` - Create collection
- `update_collection_metadata` - Update collection
- `delete_collection` - Delete collection
- `add_collection_item` / `update_collection_item` / `delete_collection_item` - Manage items

**Suggested CLI group: `hf collections`**

#### 2.4 Space Management
Spaces have many management features not exposed:
- `pause_space` / `restart_space` - Control Space runtime
- `duplicate_space` - Duplicate a Space
- `get_space_runtime` - Get runtime info
- `request_space_hardware` - Request hardware
- `set_space_sleep_time` - Configure sleep timeout
- `add_space_secret` / `delete_space_secret` - Manage secrets
- `add_space_variable` / `delete_space_variable` - Manage variables
- `request_space_storage` / `delete_space_storage` - Manage persistent storage

**Suggested commands: `hf spaces pause`, `hf spaces restart`, `hf spaces duplicate`, `hf spaces secrets/*`, `hf spaces variables/*`**

### Medium Priority

#### 2.5 Access Request Management
For gated repos:
- `list_pending_access_requests`
- `list_accepted_access_requests`
- `list_rejected_access_requests`
- `accept_access_request` / `reject_access_request` / `cancel_access_request`
- `grant_access`

**Suggested CLI group: `hf access` or `hf repo access`**

#### 2.6 Webhook Management
- `list_webhooks`
- `create_webhook`
- `get_webhook`
- `update_webhook`
- `enable_webhook` / `disable_webhook`
- `delete_webhook`

**Suggested CLI group: `hf webhooks`**

#### 2.7 User/Org Information
- `get_user_overview` - Get user profile
- `get_organization_overview` - Get org profile
- `list_user_followers` / `list_user_following`
- `list_organization_members` / `list_organization_followers`

**Suggested commands: `hf user info`, `hf org info`**

### Lower Priority

#### 2.8 Papers
- `list_papers`
- `paper_info`
- `list_daily_papers`

#### 2.9 Likes Management
- `unlike` - Unlike a repo (API note: `like` is intentionally not available to prevent spam)
- `list_liked_repos` - List liked repos
- `list_repo_likers` - List users who liked a repo

#### 2.10 LFS File Management
- `list_lfs_files` - List LFS files in a repo
- `permanently_delete_lfs_files` - Delete orphan LFS files

#### 2.11 Safetensors Metadata
- `get_safetensors_metadata` - Get safetensors metadata for a repo
- `parse_safetensors_file_metadata` - Parse individual file metadata

#### 2.12 Auth Check
- `auth_check` - Verify token has access to a specific repo/action

---

## 3. Issues with Current CLI

### 3.1 Inconsistent Output Formats

**Problem**: Some commands output JSON, others output plain text, making automation difficult.

| Command | Output Format |
|---------|--------------|
| `hf models ls` | JSON |
| `hf models info` | JSON |
| `hf cache ls` | Table (default), JSON, CSV |
| `hf jobs ps` | Table |
| `hf endpoints ls` | JSON |
| `hf auth whoami` | Plain text |
| `hf download` | Path (plain text) |
| `hf upload` | URL (plain text) |

**Recommendation**: 
- Add `--format` / `--output` option globally (json, table, plain, csv)
- Make JSON the default for agent-friendliness, or add `--json` flag universally

### 3.2 Inconsistent Command Naming

| Inconsistency | Examples |
|---------------|----------|
| `ls` vs `list` | `hf cache ls`, `hf models ls`, but `hf auth list`, `hf repo tag list` |
| `info` vs `describe` vs `inspect` | `hf models info`, `hf endpoints describe`, `hf jobs inspect` |
| Singular vs plural groups | `hf repo` (singular), `hf models` (plural) |

**Recommendation**: Standardize on:
- `ls` for all list operations
- `info` for getting details about a single resource
- Plural nouns for all groups (`repos`, `models`, `datasets`, `spaces`)

### 3.3 Missing Global Options

**Problem**: Common options that should be available everywhere:
- `--quiet` / `-q` - Only on some commands (download, upload)
- `--json` - Only on cache ls
- `--verbose` / `-v` - Not available
- `--dry-run` - Only on specific commands

**Recommendation**: Add these as global flags inherited by all commands.

### 3.4 Error Messages Not Machine-Readable

**Problem**: Error messages are human-readable strings, not structured.

```bash
$ hf models info nonexistent/model
Model nonexistent/model not found.
```

**Recommendation**: With `--json` flag, errors should also be JSON:
```json
{"error": "RepositoryNotFoundError", "message": "Model nonexistent/model not found.", "code": 404}
```

### 3.5 Exit Codes

**Current state**: Commands do use non-zero exit codes for errors (good!), but exit codes aren't documented and don't follow a consistent scheme.

**Recommendation**: Document exit codes:
- 0: Success
- 1: General error
- 2: Invalid arguments
- 64-78: Following sysexits.h conventions

---

## 4. Suggestions for New Options/Arguments

### 4.1 `hf download`

Current gaps:
- **`--progress`**: Option to show/hide progress (currently uses `--quiet` which does more than just progress)
- **`--verify`**: Verify checksums after download
- **`--output-path`**: Alternative to `--local-dir` that's more intuitive
- **`--parallel` / `--no-parallel`**: Control parallel downloads explicitly

### 4.2 `hf upload`

Current gaps:
- **`--dry-run`**: Show what would be uploaded without uploading
- **`--checksum`**: Verify checksums before/after upload
- **`--retries`**: Number of retry attempts

### 4.3 `hf models/datasets/spaces ls`

Current gaps:
- **`--direction`**: Sort direction (asc/desc) - API supports it
- **`--full`**: Return all fields (API has `full` param)
- **`--linked`**: Filter by linked repos (API supports)
- **`--gated`**: Filter by gated status
- **`--tags`**: Filter by multiple tags (currently only `--filter`)
- **`--pipeline-tag`**: Filter models by pipeline tag

### 4.4 `hf repo create`

Current gaps:
- **`--clone`**: Clone the repo after creating it
- **`--template`**: Create from template repo
- **`--license`**: Set license during creation

### 4.5 `hf cache ls`

Add:
- **`--repo-id`**: Filter by specific repo ID pattern
- **`--path`**: Show full cache paths

### 4.6 `hf auth`

Add:
- **`--json`**: Machine-readable output for `whoami`
- **`--check`**: Verify token is valid and has specific permissions

---

## 5. UX Improvements

### 5.1 Help Text Improvements

**Current**: Help text is functional but could be more discoverable.

**Suggestions**:
- Add examples in help text for each command
- Add `--examples` flag to show common use cases
- Group related options together in help output
- Add "See Also" sections pointing to related commands

### 5.2 Interactive Mode Improvements

**Current**: `hf auth switch` has interactive selection, but it's basic.

**Suggestions**:
- Use arrow keys for selection (if terminal supports)
- Add `--interactive` / `-i` flag to enable interactive mode for other commands
- Add `hf interactive` or `hf shell` for a REPL-like experience

### 5.3 Confirmation Prompts

**Current**: Some destructive operations ask for confirmation (e.g., `hf endpoints delete`), others don't.

**Suggestions**:
- Standardize: All destructive operations should confirm unless `--yes`/`-y` is passed
- Add `--force` as an alias for `--yes` for common muscle memory
- Destructive commands: `delete`, `rm`, `prune`, `move`

### 5.4 Progress and Status Reporting

**Suggestions**:
- Standardize progress bar style across all commands
- Add `--progress=none|bar|dots|percentage` option
- For long operations, show ETA and throughput
- Support `--status-fd=N` to write status updates to a specific file descriptor

### 5.5 Configuration File Support

**Current**: Relies entirely on environment variables and command-line flags.

**Suggestions**:
- Support `~/.config/hf/config.toml` or `~/.hfconfig`
- Allow setting defaults for common options
- Example config:
```toml
[defaults]
format = "json"
quiet = false
token = "hf_xxx"  # Or reference to stored token name

[download]
local_dir = "~/hf_models"
max_workers = 16
```

### 5.6 Logging Improvements

**Current**: Uses `logging.set_verbosity_info()` when not in debug mode.

**Suggestions**:
- Add `--log-level` option (debug, info, warning, error)
- Add `--log-file` to redirect logs
- Separate progress output from log output (logs to stderr, progress to stderr, results to stdout)
- Support structured logging with `--log-format=json`

---

## 6. Agent-Specific Improvements

For AI agents discovering and using the CLI:

### 6.1 Discoverability

- **`hf --list-commands`**: Flat list of all available commands (useful for agents)
- **`hf <command> --describe`**: Machine-readable command description
- **`hf --schema`**: Output JSON schema of all commands and their parameters
- **`hf --capabilities`**: List what the CLI can do in structured format

### 6.2 Predictable Output

- **Always** include a `--json` option for machine-readable output
- **Standard envelope**: 
```json
{
  "success": true,
  "data": {...},
  "warnings": [],
  "meta": {"took_ms": 123}
}
```

### 6.3 Idempotent Operations

- Commands should be idempotent where possible
- `hf repo create --exist-ok` (already exists ✓)
- `hf download` (already has idempotent caching ✓)
- Add `--if-exists=skip|error|update` for more control

### 6.4 Batch Operations

- Add support for batch operations from file:
  ```bash
  hf batch --file operations.json
  ```
- This would allow agents to queue multiple operations

### 6.5 Output Path Guarantees

- Downloads should always output the exact path downloaded
- Uploads should always output the exact URL/commit
- Add `--output-only` to output ONLY the result path/URL (no progress, no messages)

---

## 7. Specific Command Improvements

### 7.1 `hf download`

```bash
# Should support:
hf download org/model --verify          # Verify checksums
hf download org/model --json            # Output JSON with paths
hf download org/model --if-exists=skip  # Skip if already downloaded
```

### 7.2 `hf upload`

```bash
# Should support:
hf upload org/model . --dry-run         # Show what would be uploaded
hf upload org/model . --json            # Output JSON with commit URL
hf upload org/model . --verify          # Verify after upload
```

### 7.3 `hf repo`

```bash
# Missing commands:
hf repo exists org/model                # Check if repo exists
hf repo tree org/model                  # List files in tree format
hf repo commits org/model               # List commits
hf repo refs org/model                  # List branches and tags
hf repo clone org/model ./local         # Clone a repo
```

### 7.4 `hf spaces`

```bash
# Missing commands:
hf spaces pause org/space               # Pause a space
hf spaces restart org/space             # Restart a space
hf spaces duplicate org/space           # Duplicate a space
hf spaces hardware org/space            # Get/set hardware
hf spaces logs org/space                # Stream logs
hf spaces secrets ls org/space          # List secrets
hf spaces secrets add org/space KEY VAL # Add secret
hf spaces variables ls org/space        # List variables
```

### 7.5 New `hf pr` / `hf discussions` group

```bash
hf pr ls org/model                      # List PRs
hf pr create org/model "title"          # Create PR
hf pr merge org/model 42                # Merge PR #42
hf pr comment org/model 42 "message"    # Comment on PR
hf pr close org/model 42                # Close PR
hf discussions ls org/model             # List discussions
hf discussions create org/model "title" # Create discussion
```

### 7.6 New `hf collections` group

```bash
hf collections ls                       # List your collections
hf collections create "name"            # Create collection
hf collections add <slug> org/model     # Add item
hf collections rm <slug> org/model      # Remove item
hf collections info <slug>              # Get collection details
hf collections delete <slug>            # Delete collection
```

---

## 8. Documentation Suggestions

- Add a "CLI for Agents" guide explaining:
  - How to use `--json` output
  - Exit codes and their meanings
  - Idempotency guarantees
  - Rate limiting considerations
- Add shell completion instructions prominently
- Create a quick reference card / cheatsheet

---

## Summary of Recommendations by Priority

### Critical (Agent Usability)
1. Add `--json` flag to ALL commands
2. Standardize error output format
3. Document exit codes
4. Add `hf repo tree` and `hf repo exists`

### High Priority
5. Add `hf discussions/pr` commands
6. Add `hf collections` commands
7. Add Space management commands (`pause`, `restart`, `secrets`, `variables`)
8. Standardize command naming (`ls` vs `list`, `info` vs `describe`)

### Medium Priority
9. Add `hf webhooks` commands
10. Add `hf access` commands for gated repos
11. Add `--dry-run` to upload commands
12. Add `--verify` to download/upload
13. Configuration file support

### Nice to Have
14. Add `hf papers` commands
15. Add batch operation support
16. Add `hf user/org info` commands
17. Interactive shell mode

---

*This assessment was generated by comparing `hf_api.py` (173 public methods) with the current CLI implementation (17 CLI modules).*
