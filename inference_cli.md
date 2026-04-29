# Inference CLI — Design Document

> Adding `hf inference` commands to the `huggingface_hub` CLI, wrapping the existing `InferenceClient` (29 tasks, 20+ providers).

---

## 1. Landscape — How Existing CLIs Handle Inference

### 1.1 Simon Willison's `llm`

The gold standard for CLI-based LLM inference. Key design choices:

| Aspect | Design | Why it works |
|--------|--------|--------------|
| Invocation | `llm "prompt"` | Zero-friction, no subcommand needed for the common case |
| Conversation | `llm -c "follow-up"` | Continues last conversation — no session IDs to manage |
| Stdin | `cat file \| llm -s "explain"` | Full Unix composability |
| Streaming | On by default | Immediate feedback; `--no-stream` to disable |
| Model selection | `llm -m gpt-4o "prompt"` | Short aliases, configurable defaults |
| Structured output | `--schema 'name, bio, age int'` | Concise DSL instead of verbose JSON Schema |
| Logging | SQLite database | Every interaction is queryable — great for auditing |
| Providers | Plugin system | `llm install llm-anthropic` adds any provider |
| Output | Plain text to stdout | Pipeable; `--json` for machine-readable output |

**Strengths**: Simplicity-first design. One command covers 90% of use cases. Conversation state is effortless. Full pipe support makes it a first-class Unix citizen.

**Limitations**: Focused almost exclusively on text. No built-in image generation, audio, embeddings, or other modalities. The plugin system adds flexibility but fragments the experience.

### 1.2 Ollama

Local-first inference with a Docker-like UX.

| Aspect | Design |
|--------|--------|
| Invocation | `ollama run llama3 "prompt"` |
| Interactive | Drops into a REPL if no prompt given |
| Model management | `ollama pull`, `ollama list`, `ollama rm` |
| Customization | `Modelfile` (Dockerfile-like) for system prompts, parameters |
| Multimodal | `ollama run llava "describe this image" --image photo.jpg` |

**Strengths**: The "just works" experience. `ollama run model` is memorable and intuitive. The interactive REPL mode is great for exploration. Model management is built-in.

**Limitations**: Local-only (no routing to cloud providers). Limited to text and vision — no audio, no embeddings CLI. No structured output support.

### 1.3 OpenAI CLI

API-mirroring approach: the CLI mirrors the REST API structure.

```
openai api chat.completions.create -m gpt-4 -g user "Hello"
```

**Strengths**: Full API coverage — anything the API can do, the CLI can do.

**Limitations**: Terrible UX for humans. The `api chat.completions.create` path is verbose. `-g user "content"` for messages is awkward. Designed for debugging, not daily use. No streaming in early versions. No conversation state. Not pipeable.

### 1.4 Replicate CLI

```
replicate run stability-ai/sdxl "a photo of an astronaut"
```

| Aspect | Design |
|--------|--------|
| Invocation | `replicate run <model> [prompt]` |
| Output | Auto-saves files; `--web` opens in browser |
| Chaining | `{{.output[0]}}` template syntax for piping between models |
| Streaming | Supported for text models |

**Strengths**: Clean `run` verb. Multimodal-aware (images saved to files automatically). Output chaining between models is innovative.

**Limitations**: Model IDs are verbose (`owner/model:version`). Limited parameter control from CLI. No conversation state.

### 1.5 Together AI CLI

```
together chat.completions.create --model meta-llama/... --message "user:Hello"
```

**Strengths**: Full API coverage, streaming support.

**Limitations**: Same API-mirroring anti-pattern as OpenAI. Verbose, not human-friendly.

### 1.6 MLX-LM (Apple)

```
mlx_lm.generate --model mlx-community/Llama-3-8B --prompt "hello"
mlx_lm.server --model mlx-community/Llama-3-8B   # starts OpenAI-compatible server
```

**Strengths**: Prompt caching (`--cache-prompt`) saves expensive context to disk. Clean separation of generate vs serve.

**Limitations**: Local-only. No conversation support. Limited to text generation.

### 1.7 Summary of Patterns

| Pattern | Good Examples | Anti-Pattern |
|---------|--------------|--------------|
| `tool "prompt"` for common case | llm, ollama | OpenAI's `api chat.completions.create` |
| Streaming by default | llm, ollama | No streaming (blank terminal) |
| Stdin piping | llm | Most others ignore stdin |
| Conversation with `-c` | llm | Requiring session IDs or no support |
| Auto-save binary outputs | replicate | Dumping binary to stdout |
| Model aliases / defaults | llm, ollama | Requiring full model IDs every time |
| Interactive REPL fallback | ollama | Interactive-only (can't script) |

---

## 2. Key UX Decisions

### 2.1 Command Structure — Flat vs Task-Based

**Option A: Single `hf inference run` command (recommended)**

```bash
# Text generation (most common use case) — shorthand
hf inference run meta-llama/Llama-3-8B "What is the capital of France?"

# Explicit task flag for non-default tasks
hf inference run black-forest-labs/FLUX.1-dev "a cat in space" --task text-to-image

# Stdin piping
cat article.txt | hf inference run facebook/bart-large-cnn --task summarization
```

Rationale: Mirrors `ollama run` and `replicate run`. A single entry point keeps the command memorable. The `--task` flag (auto-inferred from the model's metadata on the Hub when possible) handles the long tail of tasks.

**Option B: Task subcommands**

```bash
hf inference chat meta-llama/Llama-3-8B "Hello"
hf inference text-to-image black-forest-labs/FLUX.1-dev "a cat in space"
hf inference transcribe openai/whisper-large-v3 audio.mp3
hf inference embed BAAI/bge-small-en-v1.5 "some text"
```

Rationale: More discoverable. `hf inference --help` lists all available tasks. Each task has its own tailored flags. But 29 tasks = 29 subcommands, which is overwhelming.

**Option C: Hybrid — shortcuts for top tasks + generic `run`**

```bash
# Top-level shortcuts for the 3-4 most common tasks
hf inference chat meta-llama/Llama-3-8B "Hello"
hf inference generate meta-llama/Llama-3-8B "Once upon a time"
hf inference image black-forest-labs/FLUX.1-dev "a cat in space"

# Generic run for everything else
hf inference run openai/whisper-large-v3 audio.mp3 --task automatic-speech-recognition
```

**Recommendation**: **Option C** — the hybrid approach. The common tasks deserve first-class ergonomics, while `run` covers the full 29-task surface. This follows the Pareto principle: 3-4 commands handle 90% of usage, and `run` handles the rest.

### 2.2 Chat vs Completion

The `InferenceClient` exposes both `chat_completion()` and `text_generation()`. These map to fundamentally different UX:

- **`chat`**: Conversational. Messages have roles. Streaming is expected. The user wants to *talk* to a model.
- **`generate`** (text completion): Single prompt in, text out. The user wants to *complete* text.

Both should be supported, but `chat` should be the more prominent command since chat models dominate usage.

### 2.3 Streaming

**Default: stream for text tasks, don't stream for binary tasks.**

- `hf inference chat` / `hf inference generate`: Stream by default (tokens appear as generated). `--no-stream` to buffer.
- `hf inference image` / `hf inference run --task text-to-speech`: No streaming (binary output saved to file).

This matches user expectations: you want to see text appear incrementally, but you want an image file, not a stream of bytes.

### 2.4 Input Handling

**Text input** — three sources, checked in order:
1. Positional argument: `hf inference chat model "prompt here"`
2. Stdin (if no positional prompt): `echo "prompt" | hf inference chat model`
3. Interactive REPL (if neither): `hf inference chat model` drops into a conversation

**File/binary input** — for tasks like transcription, image-to-text:
```bash
hf inference run openai/whisper-large-v3 audio.mp3
hf inference run Salesforce/blip-image-captioning-base photo.jpg
```

The first positional arg after the model is interpreted based on the task:
- Text tasks: it's the prompt
- File-consuming tasks: it's the input file path (or URL)

**Stdin piping** — essential for composability:
```bash
cat document.txt | hf inference chat model -s "Summarize this"
cat image.png | hf inference run model --task image-to-text
```

The `-s` / `--system` flag sets a system prompt, and stdin provides the user content. This mirrors `llm`'s pattern.

### 2.5 Output Handling

| Output type | Default behavior | Override |
|-------------|-----------------|----------|
| Text | Print to stdout | `--json` for structured output |
| Image | Save to file, print path | `-o filename.png` to choose path |
| Audio | Save to file, print path | `-o filename.wav` to choose path |
| Video | Save to file, print path | `-o filename.mp4` to choose path |
| Embeddings | Print as JSON array | `--json` for full response |
| Classifications | Print as table | `--json` for full response |

Binary outputs should NEVER be dumped to stdout by default (unlike `curl`). Auto-generated filenames should be descriptive: `inference_output_20240315_143022.png`.

For text outputs, `--json` returns the full API response (with usage stats, finish reason, etc.) instead of just the generated text.

### 2.6 Model Selection & Defaults

```bash
# Explicit model (always works)
hf inference chat meta-llama/Llama-3-8B "Hello"

# Provider routing
hf inference chat meta-llama/Llama-3-8B "Hello" --provider together

# No model specified — use a sensible recommended default for the task
hf inference chat "Hello"
```

When no model is provided, the CLI could:
- Use a recommended model for the task (the Hub already has this metadata)
- Let the provider's "auto" routing pick one
- Error and ask the user to specify

**Recommendation**: Allow omitting the model and rely on auto-routing (provider="auto") to select a recommended model. This makes the zero-config experience `hf inference chat "Hello"` work out of the box for logged-in users. A `--model` / `-m` flag can also be used as an alternative to the positional argument.

### 2.7 Conversation State

For `hf inference chat`:

```bash
# Start a conversation
hf inference chat model "What is Python?"

# Continue the last conversation
hf inference chat model -c "Tell me more"
```

`llm`'s `-c` (continue) flag is the simplest approach. Implementation: store the last conversation in a local file (`~/.cache/huggingface/cli/last_conversation.json`). No need for a full conversation database — keep it simple for v1.

**Considerations**:
- `-c` continues the last conversation (regardless of model)
- `--conversation ID` could be added later for named conversations
- The stored conversation is just the messages array — easy to serialize

### 2.8 Authentication

Leverage the existing CLI auth system entirely:

```bash
# Uses stored HF token (from `hf auth login`)
hf inference chat model "Hello"

# Explicit token override
hf inference chat model "Hello" --token hf_xxx

# Provider-specific API key via environment variable
HF_TOKEN=hf_xxx hf inference chat model "Hello"
```

No new auth concepts needed. The `InferenceClient` already handles token resolution and provider routing based on the token type.

### 2.9 Multimodal Inputs in Chat

Chat models increasingly support images, audio, and documents. The CLI should handle this:

```bash
# Attach an image to a chat message
hf inference chat model "What's in this image?" -a image.jpg

# Attach multiple files
hf inference chat model "Compare these" -a photo1.jpg -a photo2.jpg

# Attach a URL
hf inference chat model "Describe this" -a https://example.com/image.png
```

The `-a` / `--attach` flag (inspired by `llm`'s `-a` flag) adds files as multimodal content in the user message. The CLI determines the MIME type from the file extension and constructs the appropriate multimodal message format.

---

## 3. Proposed Command Structure

### 3.1 Command Group

```
hf inference
├── chat        # Chat completion (conversational)
├── generate    # Text generation (completion)
├── image       # Text-to-image generation
├── run         # Generic: any of the 29 tasks
└── [future: transcribe, speak, embed, ...]
```

### 3.2 `hf inference chat`

The flagship command. Conversational text generation with streaming.

```
hf inference chat [MODEL] [PROMPT] [OPTIONS]

Arguments:
  MODEL                Model ID on the Hub (optional if default configured)
  PROMPT               The message to send (reads from stdin if omitted)

Options:
  -s, --system TEXT    System prompt
  -c, --continue       Continue the last conversation
  -a, --attach FILE    Attach a file (image, audio, document) — repeatable
  -m, --model TEXT     Alternative to positional MODEL argument
  --provider TEXT      Inference provider (auto, together, openai, ...)
  --max-tokens INT     Maximum tokens to generate
  --temperature FLOAT  Sampling temperature (0.0–2.0)
  --top-p FLOAT        Nucleus sampling threshold
  --seed INT           Random seed for reproducibility
  --no-stream          Buffer full response instead of streaming
  --json               Output full JSON response (with usage stats)
  --token TEXT         HF token override

Examples:
  $ hf inference chat "What is the meaning of life?"
  $ hf inference chat meta-llama/Llama-3-8B "Hello" --provider together
  $ cat essay.txt | hf inference chat -s "Grade this essay"
  $ hf inference chat model "What's in this image?" -a photo.jpg
  $ hf inference chat model -c "Tell me more about that"
```

Interactive REPL mode when invoked with no prompt and no stdin:
```
$ hf inference chat meta-llama/Llama-3-8B
> Hello!
Hello! How can I help you today?
> What is Python?
Python is a programming language...
> /quit
```

### 3.3 `hf inference generate`

Text completion (non-conversational).

```
hf inference generate [MODEL] [PROMPT] [OPTIONS]

Options:
  --max-tokens INT     Maximum tokens to generate
  --temperature FLOAT  Sampling temperature
  --no-stream          Buffer full response
  --json               Output full JSON response
  --token TEXT         HF token override
  --provider TEXT      Inference provider

Examples:
  $ hf inference generate bigcode/starcoder2 "def fibonacci(n):"
  $ echo "Once upon a time" | hf inference generate model
```

### 3.4 `hf inference image`

Text-to-image generation.

```
hf inference image [MODEL] PROMPT [OPTIONS]

Options:
  -o, --output FILE        Output file path (default: auto-generated)
  --width INT              Image width
  --height INT             Image height
  --negative-prompt TEXT   What to avoid in the image
  --guidance-scale FLOAT   Guidance scale
  --num-steps INT          Number of inference steps
  --seed INT               Random seed
  --provider TEXT          Inference provider
  --token TEXT             HF token override

Examples:
  $ hf inference image "a cat in space"
  $ hf inference image black-forest-labs/FLUX.1-dev "cyberpunk city" -o city.png
  $ hf inference image model "portrait" --width 1024 --height 1024 --seed 42
```

### 3.5 `hf inference run`

Generic command for all 29 tasks.

```
hf inference run MODEL INPUT [OPTIONS]

Arguments:
  MODEL              Model ID on the Hub
  INPUT              Text prompt, file path, or URL (depends on task)

Options:
  --task TEXT         Task name (auto-detected from model if possible)
  -o, --output FILE  Output file path (for binary outputs)
  --json             Output as JSON
  --provider TEXT     Inference provider
  --token TEXT       HF token override
  -e, --extra KEY=VAL Extra parameters (repeatable, passed as extra_body)

Examples:
  # Transcription
  $ hf inference run openai/whisper-large-v3 meeting.mp3 --task automatic-speech-recognition

  # Summarization
  $ cat article.txt | hf inference run facebook/bart-large-cnn --task summarization

  # Text-to-speech
  $ hf inference run facebook/mms-tts-eng "Hello world" -o greeting.wav

  # Embeddings
  $ hf inference run BAAI/bge-small-en-v1.5 "some text" --task feature-extraction

  # Image classification
  $ hf inference run google/vit-base-patch16-224 photo.jpg --task image-classification

  # Translation
  $ echo "Bonjour le monde" | hf inference run Helsinki-NLP/opus-mt-fr-en --task translation
```

The `--extra` / `-e` flag passes arbitrary key-value pairs to `extra_body`, allowing provider-specific parameters without bloating the CLI flags:
```bash
hf inference run model "prompt" -e num_inference_steps=50 -e guidance_scale=7.5
```

---

## 4. Integration with Existing CLI

### 4.1 Registration

In `hf.py`, add:
```python
from huggingface_hub.cli.inference import inference_cli
app.add_typer(inference_cli, name="inference | infer")
```

The alias `infer` provides a shorter alternative: `hf infer chat "Hello"`.

### 4.2 Shared Options to Reuse

From `_cli_utils.py`:
- `TokenOpt` — for `--token`
- `FormatOpt` / `OutputFormat` — for `--json` / `--format` on list-like outputs
- `generate_epilog()` — for example generation in help text
- Error handling in `_errors.py` — add `InferenceTimeoutError` handler

### 4.3 New Shared Options to Define

```python
ModelArg = Annotated[Optional[str], typer.Argument(help="Model ID on the Hub")]
ProviderOpt = Annotated[Optional[str], typer.Option("--provider", help="Inference provider")]
```

### 4.4 Error Handling Additions

Add to `_errors.py`:
- `InferenceTimeoutError` → "Inference request timed out. The model may be loading — try again in a few seconds."
- `HfHubHTTPError` with 422 → "Invalid input for model. Check --task and input format."

### 4.5 Output Binary Files

New utility needed: a helper to auto-generate output file paths and save binary content (images, audio, video), printing the saved path to stdout. Similar to how `download.py` prints the downloaded file path.

---

## 5. Open Questions

### 5.1 Should `hf inference chat` support tool use?

The `InferenceClient.chat_completion()` supports `tools` and `tool_choice`. Exposing this from the CLI is complex (how do you define tool schemas on the command line?). **Suggestion**: defer to v2. For v1, focus on text-in/text-out.

### 5.2 How to handle model auto-detection for tasks?

When the user runs `hf inference run model input`, can we auto-detect the task from the model's metadata on the Hub (via `model_info().pipeline_tag`)? This would eliminate the need for `--task` in most cases. **Suggestion**: yes, auto-detect with `--task` as override.

### 5.3 Should there be a config command?

```bash
hf inference config set default-model meta-llama/Llama-3-8B
hf inference config set default-provider together
```

`llm` has `llm models default gpt-4o-mini`. This reduces typing but adds complexity. **Suggestion**: support environment variables first (`HF_INFERENCE_MODEL`, `HF_INFERENCE_PROVIDER`), and add a config command later if users ask for it.

### 5.4 Interactive REPL scope

Should the REPL mode for `hf inference chat` support slash commands (`/system`, `/model`, `/clear`, `/save`)? **Suggestion**: v1 should have a minimal REPL (just type messages, Ctrl+C to exit). Slash commands can come in v2.

### 5.5 Conversation history persistence

How long should conversations be kept? Where? **Suggestion**: keep only the last conversation for `-c`. Store in `~/.cache/huggingface/cli/last_conversation.json`. More sophisticated history (named conversations, SQLite) can come later.

### 5.6 What about `hf chat` as a top-level alias?

Since chat is the most common inference use case, should there be a shortcut?

```bash
hf chat "Hello"           # alias for: hf inference chat "Hello"
```

This is aggressive but matches the trend (Ollama's top-level is `ollama run`). The existing CLI already has top-level commands (`hf download`, `hf upload`). **Suggestion**: add `hf chat` as an alias if the command proves popular.

### 5.7 Structured output (`--schema`)

`llm`'s `--schema 'name, bio, age int'` is compelling. Could be mapped to `response_format` in `chat_completion()`. **Suggestion**: support `--response-format json` for v1 (asks the model to respond in JSON). A schema DSL can come later.

### 5.8 Batch processing

Should the CLI support processing multiple inputs?

```bash
# Process multiple images
hf inference run model *.jpg --task image-classification

# Process lines from a file
cat prompts.txt | hf inference chat model --batch
```

**Suggestion**: defer to v2. Single-input is complex enough for v1.

---

## 6. Recommended Implementation Priority

**v1 — Core** (ship first):
1. `hf inference chat` — with streaming, stdin, `-s` system prompt, `--provider`
2. `hf inference run` — generic task runner with `--task` and auto-detection
3. `hf inference image` — text-to-image with file output

**v2 — Polish**:
4. `hf inference generate` — text completion
5. `-c` conversation continuation
6. `-a` multimodal attachments
7. Interactive REPL mode for chat
8. `hf chat` top-level alias

**v3 — Power features**:
9. `--schema` / structured output DSL
10. Named conversations
11. Batch processing
12. Additional shortcut commands (`transcribe`, `embed`, `speak`)
