# Commit Critic

AI-powered terminal tool that analyzes commit message quality and helps write better commits.

Uses a two-phase **diff-aware** approach: the LLM doesn't just check message formatting -- it compares messages against actual code changes to suggest contextually accurate improvements.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
# or for Claude:
export ANTHROPIC_API_KEY=sk-ant-...
# or for Gemini:
export GEMINI_API_KEY=...
```

On first run, an interactive model picker lets you choose and optionally save your model to `.env`. You can also set it directly:

```bash
export LITELLM_MODEL=claude-sonnet-4-20250514
```

Any [litellm-supported provider](https://docs.litellm.ai/docs/providers) works.

## Usage

### Analyze commits

```bash
# Analyze last 50 commits on main
python commit_critic.py --analyze

# Analyze last 20 commits on a specific branch
python commit_critic.py --analyze -n 20 -b develop

# Analyze a remote repository
python commit_critic.py --analyze --url="https://github.com/user/repo"

# Filter by author
python commit_critic.py --analyze --author="ray"

# Use a specific model
python commit_critic.py --analyze -m gpt-4o
```

<details>
<summary>Demo: --analyze</summary>

```
$ python commit_critic.py --analyze -n 5

[gpt-4o-mini] my-project | main

Analyzing 5 commits...

╭──────────────── COMMITS THAT NEED WORK ─────────────────╮
│ Commit: "fix"                                           │
│ Score: 1/10                                             │
│ Issue: No context about what was fixed or why           │
│ Better: "fix(api): handle null response from /users"    │
│                                                         │
│ Commit: "update stuff"                                  │
│ Score: 2/10                                             │
│ Issue: Vague description with no scope or detail        │
│ Better: "refactor(auth): simplify token refresh logic"  │
│                                                         │
│ Commit: "wip"                                           │
│ Score: 1/10                                             │
│ Issue: Placeholder message with no useful information   │
│ Better: "feat(cart): add quantity validation"            │
╰─────────────────────────────────────────────────────────╯
╭──────────────── WELL-WRITTEN COMMITS ───────────────────╮
│ Commit: "feat(search): add fuzzy matching with Fuse.js" │
│ Score: 9/10                                             │
│ Why it's good: Clear type, scope, and implementation    │
│                                                         │
│ Commit: "fix(db): add index on users.email for login"   │
│ Score: 8/10                                             │
│ Why it's good: Specific scope with measurable change    │
╰─────────────────────────────────────────────────────────╯
╭──────────────────── YOUR STATS ─────────────────────────╮
│ Average score: 4.2/10                                   │
│ Vague commits: 3 (60%)                                  │
│ One-word commits: 2 (40%)                               │
│                                                         │
│ Score distribution:                                     │
│    1-3  ████████████████████  3                          │
│    4-6                        0                          │
│    7-8  █████████████         2                          │
│   9-10                        0                          │
│ Tokens used: 1,247                                      │
╰─────────────────────────────────────────────────────────╯
```

</details>

### Write a commit message

```bash
# Stage your changes first
git add -p

# Get an AI-suggested commit message
python commit_critic.py --write
```

<details>
<summary>Demo: --write</summary>

```
$ git add src/auth.py src/middleware.py
$ python commit_critic.py --write

[gpt-4o-mini] my-project | feat/token-refresh

Analyzing staged changes... (2 files changed, 47 insertions(+), 12 deletions(-))

Changes detected:
  - Add refresh logic triggered 5 min before expiry
  - Extract token validation into reusable middleware
  - Add REFRESH_WINDOW_SECONDS env config

╭──────────── Suggested commit message ───────────────────╮
│ feat(auth): add automatic token refresh                 │
│                                                         │
│ - Add refresh logic triggered 5 min before expiry       │
│ - Extract token validation into reusable middleware     │
│ - Add REFRESH_WINDOW_SECONDS env config (default: 300)  │
╰─────────────────────────────────────────────────────────╯
Tokens used: 834

Press Enter to accept, or type your own message:
Committed! [feat/token-refresh abc1234] feat(auth): add automatic token refresh
```

</details>

The `--write` flag reads your staged diff, generates a conventional commit message, and lets you accept or override it before committing.

### CLI flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--analyze` | | Analyze commit message quality | |
| `--write` | | Suggest a commit message for staged changes | |
| `--count` | `-n` | Number of commits to analyze | 50 |
| `--url` | `-u` | Remote repo URL to analyze | |
| `--branch` | `-b` | Branch to analyze | main |
| `--author` | `-a` | Filter commits by author | all authors |
| `--model` | `-m` | LLM model to use | interactive picker |

Positional syntax also works: `python commit_critic.py analyze` / `python commit_critic.py write`.

## How It Works

### Two-Phase Diff-Aware Analysis

**Phase 1 -- Batch Score** (1 LLM call): All commit messages are scored on a 1-10 scale based on clarity, scope, and conventional commit format.

**Phase 2 -- Diff Context** (1 LLM call): Commits scoring < 7 are re-analyzed with their actual `git show --stat` output. The LLM suggests messages that match what the code actually changed, not just generic format improvements.

This matters because "fix bug" might be acceptable if the diff shows a one-line typo fix, but terrible if it touches 15 files across 3 modules.

### README-Based Project Context

When available, the repo's README is read (up to 1500 chars) and prepended to every LLM prompt. This gives the model project-specific context -- it can score "add caching" differently for a database library vs. a frontend app, and suggest scope names that match the project's actual modules.

### Stats Computed in Python

Average score, vague commit count, and one-word commit count are calculated deterministically in Python -- never delegated to the LLM. This avoids arithmetic hallucinations.

### Context Management

- Large diffs are truncated with file stats always preserved (small, high-signal)
- Phase 2 uses deterministic index mapping (never trusts LLM-returned identifiers)
- Token usage is tracked and displayed for every operation
- Remote repos are shallow-cloned to minimize bandwidth
- Repo README is read and injected as project context for more accurate scoring

### Security

- Branch names and URLs are validated to prevent option injection
- Local file paths (`file://`, `/`) are rejected for remote clone
- Model names are sanitized before writing to `.env`

## Architecture

Single-file design (`commit_critic.py`, ~740 lines):

| Section | Purpose |
|---------|---------|
| Configuration | Interactive model picker, `.env` persistence |
| Data models | Pydantic models for structured LLM output |
| Git operations | Log parsing, diff extraction, shallow clone, README reading, validation |
| LLM interaction | Two-phase analysis with README context, commit suggestion, retry logic |
| Output formatting | Rich panels for terminal display |
| CLI commands | Typer `--analyze` and `--write` with positional support |

## Dependencies

- **litellm** -- Provider-agnostic LLM calls (OpenAI, Anthropic, Google, etc.)
- **typer** -- CLI framework
- **pydantic** -- Structured output validation
- **rich** -- Terminal formatting
- **python-dotenv** -- `.env` file support

## Tests

```bash
pytest test_commit_critic.py -v
```

26 tests covering models, git parsing, README reading, LLM integration (mocked), output formatting, histogram, and author filtering.
