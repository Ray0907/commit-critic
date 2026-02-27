# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                      # Install dependencies
uv run commit-critic --analyze               # Run the tool (analyze mode)
uv run commit-critic --write                 # Run the tool (write mode)
uv run pytest test_commit_critic.py -v       # Run all tests
uv run pytest test_commit_critic.py -v -k "test_name"  # Run a single test
```

## Architecture

Single-file CLI tool (`commit_critic.py`, ~740 lines) that uses LLMs to analyze git commit message quality and suggest better messages. Uses `litellm` for provider-agnostic LLM calls (OpenAI, Anthropic, Google).

### Two-Phase Diff-Aware Analysis

1. **Phase 1** (1 LLM call): Batch-scores all commit messages on a 1-10 scale using `callLlmParsed()` with Pydantic models for structured JSON output
2. **Phase 2** (1 LLM call): Re-analyzes commits scoring < 7 with their actual `git show --stat` diffs. Uses deterministic index mapping (never trusts LLM-returned identifiers) to map suggestions back

### Code Sections

| Section | Key functions |
|---------|--------------|
| Configuration | `resolveModel()` -- interactive model picker with `.env` persistence |
| Data models | `CommitScore`, `AnalysisResult`, `SuggestedCommit` (Pydantic) |
| Git operations | `getCommits()`, `getCommitDiff()`, `getStagedDiff()`, `cloneShallow()` |
| LLM interaction | `callLlm()`, `callLlmParsed()`, `analyzeCommits()`, `suggestCommitMessage()` |
| Output | `renderAnalysis()`, `_buildHistogram()` (Rich panels) |
| CLI | Typer app with `--analyze` / `--write` flags and positional syntax |

### Key Design Decisions

- Stats (average, vague count, one-word count) are computed deterministically in Python, never delegated to the LLM
- README is injected as project context into every LLM prompt (up to 1500 chars) for project-aware scoring
- Large diffs are truncated to 12K chars but file stats are always preserved
- `litellm.drop_params = True` ensures compatibility across providers
- Git log uses ASCII record separator (`\x1e`) for reliable parsing

### Testing

Tests mock `subprocess.run` for git operations and `completion` for LLM calls. The `set_model` autouse fixture sets `commit_critic._model = "test-model"` to bypass the interactive model picker.
