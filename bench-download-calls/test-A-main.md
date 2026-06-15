# Test A — baseline (`main`)

Totals (cold-v1 / warm / PR-cold / warm): **small repo 504 / 1 / 403 / 1** · **large repo 5007 / 4 / 4006 / 4**.

## Small repo (201 files)

### run1 — v1 cold: 504 calls

| # | Method | Status | Endpoint |
|---|--------|--------|----------|
| 1 | GET | 200 | `/api/models/{repo}/revision/v1` (repo_info) |
| 101 | HEAD | 307 | `/{repo}/resolve/{sha}/{file}` (regular files + .gitattributes) |
| 101 | HEAD | 200 | `/api/resolve-cache/models/{repo}/{sha}/{file}` (redirect follow-up) |
| 101 | GET | 200 | `/api/resolve-cache/models/{repo}/{sha}/{file}` (content) |
| 100 | HEAD | 302 | `/{repo}/resolve/{sha}/{file}` (xet files; 302 carries metadata, not followed) |
| 100 | GET | 200 | `/api/models/{repo}/xet-read-token/{sha}` *(hf_xet, ignorable per scope)* |

- **run2 — v1 warm: 1 call** — `GET /revision/v1` only. The commit-hash shortcut in `hf_hub_download` (pointer exists → return) makes warm same-commit runs optimal.
- **run3 — PR cold: 403 calls** — same shape as run1, but only the 50 changed text files are GET-downloaded and only the 50 changed xet files trigger a token call. **The 201 HEAD chains happen regardless of whether the file changed** — unchanged files still cost 2 HEADs (regular) or 1 HEAD (xet) each just to discover they're already cached.
- **run4 — PR warm: 1 call.**

Here `repo_info.siblings` (≤1000 files) supplies the file list — **no `/tree` call**.

## Large repo (2001 files) — crosses the 1000-file threshold

### run1 — v1 cold: 5007 calls

| # | Method | Status | Endpoint |
|---|--------|--------|----------|
| 1 | GET | 200 | `/api/models/{repo}/revision/v1` |
| 3 | GET | 200 | `/api/models/{repo}/tree/{rev}` (paginated, 1000/page → 3 pages) |
| 1001 | HEAD | 307 | `/{repo}/resolve/{sha}/{file}` (regular) |
| 1001 | HEAD | 200 | `/api/resolve-cache/models/{repo}/{sha}/{file}` |
| 1001 | GET | 200 | `/api/resolve-cache/models/{repo}/{sha}/{file}` (content) |
| 1000 | HEAD | 302 | `/{repo}/resolve/{sha}/{file}` (xet) |
| 1000 | GET | 200 | `/api/models/{repo}/xet-read-token/{sha}` *(hf_xet)* |

- **run2 — v1 warm: 4 calls** — `1 revision + 3 tree`. **Not 1!** See below.
- **run3 — PR cold: 4006 calls** — 1 revision + 3 tree + 2002 regular HEAD chains + 1000 xet HEAD + 500 changed-text GET + 500 changed-xet token.
- **run4 — PR warm: 4 calls.**

## Observations

- **Cost is linear in file count, not in changed bytes**: per regular file 2 HEAD + 1 GET, per xet file 1 HEAD (+1 token when downloaded) — ~2.5 calls/file cold, at both scales (504→5007 is a clean 10×).
- **The `/api/resolve-cache` 307 redirect doubles regular-file HEAD/GET traffic** on public repos: `HEAD /resolve` never answers directly, it redirects and the target is hit again.
- **Scale-dependent regression: above 1000 files, warm runs jump from 1 to 4 calls.** When `siblings` is judged unreliable (`>LARGE_REPO_THRESHOLD`), main lists the tree via `list_repo_tree` for the file list — paginated at 1000/page (3 pages for 2001 files) — and **never caches it**, so a no-op warm re-pull still costs `1 revision + 3 tree`. Below the threshold the file list comes from `repo_info.siblings` (already fetched) so warm = 1.
- hf_xet requests **one read token per file** (1000 × `xet-read-token` on the large cold run). Out of scope for counting, but it lands on huggingface.co.
