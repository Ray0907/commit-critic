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
export LITELLM_MODEL=claude-sonnet-4-20250514
```

Any [litellm-supported provider](https://docs.litellm.ai/docs/providers) works. Default model: `gpt-4o-mini`.

## Usage

### Analyze commits

```bash
# Analyze last 50 commits in current repo
python commit_critic.py --analyze

# Analyze last 20 commits
python commit_critic.py --analyze -n 20

# Analyze a remote repository
python commit_critic.py --analyze --url="https://github.com/user/repo"
```

### Write a commit message

```bash
# Stage your changes first
git add -p

# Get an AI-suggested commit message
python commit_critic.py --write
```

The `--write` flag reads your staged diff, generates a conventional commit message, and lets you accept or override it before committing.

## How It Works

### Two-Phase Diff-Aware Analysis

**Phase 1 -- Batch Score** (1 LLM call): All commit messages are scored on a 1-10 scale based on clarity, scope, and conventional commit format.

**Phase 2 -- Diff Context** (1 LLM call): Commits scoring <= 4 are re-analyzed with their actual `git show --stat` output. The LLM suggests messages that match what the code actually changed, not just generic format improvements.

This matters because "fix bug" might be acceptable if the diff shows a one-line typo fix, but terrible if it touches 15 files across 3 modules.

### Context Management

- Large diffs are truncated with file stats always preserved (small, high-signal)
- Phase 2 uses deterministic index mapping (never trusts LLM-returned identifiers)
- Token usage is tracked and displayed for every operation
- Remote repos are shallow-cloned to minimize bandwidth

## Architecture

Single-file design (`commit_critic.py`, ~470 lines):

| Section | Purpose |
|---------|---------|
| Data models | Pydantic models for structured LLM output |
| Git operations | Log parsing, diff extraction, shallow clone |
| LLM interaction | Two-phase analysis, commit suggestion, retry logic |
| Output formatting | Rich panels for terminal display |
| CLI commands | Typer `--analyze` and `--write` flags |

## Dependencies

- **litellm** -- Provider-agnostic LLM calls (OpenAI, Anthropic, etc.)
- **typer** -- CLI framework
- **pydantic** -- Structured output validation
- **rich** -- Terminal formatting
- **python-dotenv** -- `.env` file support

## Tests

```bash
pytest test_commit_critic.py -v
```

16 tests covering models, git parsing, LLM integration (mocked), and output formatting.
