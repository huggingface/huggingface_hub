# ============================================================================
# hf sync - Complete Command Reference
# ============================================================================

# BASIC USAGE
hf sync SOURCE DEST [OPTIONS]

# SOURCE/DEST formats:
# - Local path: ./data or /absolute/path
# - Bucket path: hf://buckets/namespace/bucket_name[/prefix]
# For now, only local -> remote or remote -> local but no remote -> remote.

# Examples:
#   hf sync ./data hf://buckets/user/my-bucket
#   hf sync hf://buckets/user/my-bucket/models ./local-models

# By default (no flags):
# - Compares files by mtime + size (fast)
# - Uploads new files
# - Uploads files where local is newer
# - Skips files where remote is newer
# - NEVER deletes files (safe)

# Explicit errors (for now):
# - handle directories only (fail if local or remote path is a file)
# - fail if bucket does not exist
# - do not fail if local path does not exist (create it)

# ============================================================================
# SYNC STRATEGIES
# ============================================================================

--mirror                   # Make dest identical to source (DELETES destination files not in source)

# ============================================================================
# CONFLICT RESOLUTION
# ============================================================================

--force-upload             # Always upload/overwrite remote (ignore remote mtime and size)
--force-download           # Always download/overwrite local (ignore local mtime and size)

# For later
--no-traverse              # Don't scan destination before sync (much faster for large buckets)
--new-files-only           # Only upload new files
--existing-files-only      # Only upload existing files
--modify-window DURATION   # Treat mtime within duration as same (e.g., "2s" for clock skew)

# ============================================================================
# PLANNING
# ============================================================================

--plan FILE                # Save sync plan to JSON file for review (or most probablyJSONL ?)
--execute FILE             # Execute previously saved plan

# ============================================================================
# FILTERING
# ============================================================================

--include PATTERN          # Include files matching pattern (can specify multiple)
--exclude PATTERN          # Exclude files matching pattern (can specify multiple)
--filter-from FILE         # Read include/exclude patterns from file
# Filtering rules:
# - case sensitive
# - filters are evaluated in order, first matching rule decides
# - filter file uses "+" and "-" prefixes to include and exclude files
# - (optimization) do not list directories that are excluded
# Basically everything from https://rclone.org/filtering/

# ============================================================================
# OUTPUT & MONITORING
# ============================================================================

--verbose, -v               # Show detailed logging with reasoning
--quiet, -q                 # Minimal output

# For later
--json                      # Output progress as JSON (for scripts/CI)
--progress / --no-progress  # Show progress bars

# ============================================================================
# EXAMPLES
# ============================================================================

# 1.a Basic sync (no deletions)
hf sync ./data hf://buckets/user/my-bucket
hf sync hf://buckets/user/my-bucket ./data

# 1.b. Basic sync in subfolder
hf sync ./data hf://buckets/user/my-bucket/data/
hf sync hf://buckets/user/my-bucket/data ./data

# 2. Mirror (exact replica with deletions)
hf sync ./data hf://buckets/user/my-bucket --mirror
hf sync hf://buckets/user/my-bucket ./data --mirror

# 3. With filters
hf sync hf://buckets/user/my-bucket ./data \
  --include "*.safetensors" \
  --exclude "*.tmp"

hf sync ./data hf://buckets/user/my-bucket \
  --include "*.safetensors" \
  --exclude "*.tmp"

hf sync ./data hf://buckets/user/my-bucket \
  --include "*.parquet" \
  --include "*.csv" \
  --exclude "*.tmp" \
  --exclude ".git/**" \
  --exclude "checkpoints/**"

# 4. Force upload/download (skip mtime and size checks)
hf sync ./data hf://buckets/user/my-bucket --force-upload
hf sync hf://buckets/user/my-bucket ./data --force-download

# 5. Safe review workflow
hf sync ./data hf://buckets/user/my-bucket --plan sync-plan.json
cat sync-plan.json  # Review the plan
hf sync hf://buckets/user/my-bucket ./data --execute sync-plan.json

# ============================================================================
# SYNC PLAN JSON FORMAT (--plan)
# ============================================================================

# {
#   "source": "./data",
#   "dest": "hf://buckets/user/bucket",
#   "timestamp": "2025-01-13T10:30:00Z",
#   "operations": [
#     {
#       "action": "upload",
#       "path": "data/new_file.txt",
#       "size": 10240,
#       "reason": "new file"
#     },
#     {
#       "action": "upload",
#       "path": "data/model.bin",
#       "size": 1073741824,
#       "reason": "local newer",
#       "local_mtime": "2025-01-13T10:00:00Z",
#       "remote_mtime": "2025-01-10T15:30:00Z"
#     },
#     {
#       "action": "skip",
#       "path": "data/readme.md",
#       "reason": "identical"
#     },
#     {
#       "action": "delete",
#       "path": "data/old_file.txt",
#       "reason": "not in source (--mirror mode)"
#     }
#   ],
#   "summary": {
#     "uploads": 2,
#     "deletes": 1,
#     "skips": 1,
#     "total_size": 1073752064
#   }
# }