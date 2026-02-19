"""AI Commit Message Critic - Analyze and improve your commit messages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import TypeVar

import litellm
import typer
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

T = TypeVar("T", bound=BaseModel)

_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, ".env"))

litellm.drop_params = True

console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POPULAR_MODELS = [
    ("gpt-4o-mini",                        "OpenAI",    "OPENAI_API_KEY"),
    ("gpt-4o",                             "OpenAI",    "OPENAI_API_KEY"),
    ("claude-sonnet-4-20250514",           "Anthropic", "ANTHROPIC_API_KEY"),
    ("claude-haiku-4-5-20251001",          "Anthropic", "ANTHROPIC_API_KEY"),
    ("gemini/gemini-2.0-flash",            "Google",    "GEMINI_API_KEY"),
    ("gemini/gemini-2.5-pro-preview-05-06", "Google",   "GEMINI_API_KEY"),
]

_model: str | None = None


def resolveModel() -> str:
    """Resolve model from --model flag, env var, or interactive picker."""
    global _model
    if _model:
        return _model

    env_model = os.environ.get("LITELLM_MODEL")
    if env_model:
        _model = env_model
        return _model

    console.print("[bold]No model configured.[/bold] Pick one:\n")
    for i, (name, provider, key_env) in enumerate(POPULAR_MODELS, 1):
        key_status = "[green]set[/green]" if os.environ.get(key_env) else "[dim]not set[/dim]"
        console.print(f"  {i}) {name}  [dim]({provider}, {key_env}: {key_status})[/dim]")
    console.print("\n  Or enter any [bold]litellm[/bold]-compatible model name.")
    console.print()

    choice = Prompt.ask(
        "Enter number or model name",
        default="1",
    )

    if choice.isdigit() and 1 <= int(choice) <= len(POPULAR_MODELS):
        _model = POPULAR_MODELS[int(choice) - 1][0]
    else:
        _model = choice.strip()

    # Offer to save so user doesn't have to pick every time
    save = Prompt.ask(
        f"Save [bold]{_model}[/bold] to .env?",
        choices=["y", "n"],
        default="y",
    )
    if save == "y":
        _saveModelToEnv(_model)

    console.print(f"Using model: [bold]{_model}[/bold]\n")
    return _model


def _saveModelToEnv(model_name: str) -> None:
    """Append or update LITELLM_MODEL in .env file."""
    name_sanitized = model_name.replace("\n", "").replace("\r", "").strip()
    path_env = os.path.join(_DIR, ".env")
    line_new = f"LITELLM_MODEL={name_sanitized}\n"

    lines_existing: list[str] = []
    if os.path.exists(path_env):
        with open(path_env) as f:
            lines_existing = f.readlines()

    lines_updated = [
        line_new if line.startswith("LITELLM_MODEL=") else line
        for line in lines_existing
    ]
    if not any(line.startswith("LITELLM_MODEL=") for line in lines_existing):
        lines_updated.append(line_new)

    with open(path_env, "w") as f:
        f.writelines(lines_updated)

    console.print(f"[green]Saved to {path_env}[/green]")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CommitScore(BaseModel):
    hash: str
    message: str
    score: int
    issue: str | None = None
    suggestion: str | None = None
    praise: str | None = None


class AnalysisResult(BaseModel):
    commits: list[CommitScore]
    average_score: float = 0.0
    count_vague: int = 0
    count_one_word: int = 0


class SuggestedCommit(BaseModel):
    message: str
    changes: list[str]


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

RECORD_SEP = "\x1e"  # ASCII record separator (from pre-commit pattern)


def getGitRepoName(path: str = ".") -> str | None:
    """Get the git repo name from the remote or directory name."""
    result = subprocess.run(
        ["git", "-C", path, "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        url = result.stdout.strip().rstrip("/")
        name = url.split("/")[-1]
        return name.removesuffix(".git")
    # Fallback: top-level directory name
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return os.path.basename(result.stdout.strip())
    return None


def getGitBranch(path: str = ".") -> str | None:
    """Get the current git branch name."""
    result = subprocess.run(
        ["git", "-C", path, "branch", "--show-current"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def printStatusLine(path: str = ".", branch: str | None = None) -> None:
    """Print a Claude Code-style status line: [model] repo | branch."""
    model_name = resolveModel()
    repo_name = getGitRepoName(path) or "no-repo"
    branch_name = branch or getGitBranch(path) or "detached"
    console.print(
        f"[dim]\\[{model_name}] {repo_name} | {branch_name}[/dim]\n"
    )


def checkGitRepo(path: str = ".") -> None:
    """Exit with helpful message if not in a git repo."""
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print("[red]Not a git repository.[/red] Run this from inside a git repo.")
        raise typer.Exit(1)


def _validateBranchName(name: str) -> None:
    """Reject branch names that could inject git options."""
    if name.startswith("-"):
        console.print("[red]Invalid branch name: must not start with '-'[/red]")
        raise typer.Exit(1)


def getCommits(
    count: int,
    repo_path: str = ".",
    branch: str | None = None,
    author: str | None = None,
) -> list[dict]:
    """Extract commits from git log. Returns list of {hash, subject, body}."""
    if branch:
        _validateBranchName(branch)
    fmt = f"%H{RECORD_SEP}%s{RECORD_SEP}%b{RECORD_SEP}"
    cmd = ["git", "-C", repo_path, "log", f"--format={fmt}", "-n", str(count)]
    if branch:
        cmd.append(branch)
    if author:
        if author.startswith("-"):
            console.print("[red]Invalid author: must not start with '-'[/red]")
            raise typer.Exit(1)
        cmd.extend(["--author", author])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Error reading git log:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)

    commits = []
    for entry in result.stdout.strip().split(RECORD_SEP + "\n"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(RECORD_SEP, 2)
        if len(parts) >= 2:
            commits.append({
                "hash": parts[0][:8],
                "subject": parts[1].strip(),
                "body": parts[2].strip() if len(parts) > 2 else "",
            })
    return commits


def getCommitDiff(hash_commit: str, repo_path: str = ".") -> str:
    """Get diff stat for a specific commit (for diff-aware analysis)."""
    result = subprocess.run(
        ["git", "-C", repo_path, "show", "--stat", hash_commit],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def getStagedDiff() -> str:
    """Get the full staged diff."""
    result = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Error reading staged diff:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)
    return result.stdout


def getStagedStat() -> str:
    """Get staged diff stat summary."""
    result = subprocess.run(
        ["git", "diff", "--staged", "--stat"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validateGitUrl(url: str) -> None:
    """Reject URLs that could be used for command injection or local file access."""
    stripped = url.strip()
    if stripped.startswith("-"):
        console.print("[red]Invalid URL: must not start with '-'[/red]")
        raise typer.Exit(1)
    if stripped.startswith(("file://", "/")):
        console.print("[red]Local paths not supported. Use a remote URL.[/red]")
        raise typer.Exit(1)


def cloneShallow(url: str, count: int) -> str:
    """Shallow clone a repo to a temp directory."""
    _validateGitUrl(url)
    dir_tmp = tempfile.mkdtemp(prefix="commit_critic_")
    result = subprocess.run(
        ["git", "clone", "--depth", str(count), url, dir_tmp],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return dir_tmp
    error_msg = result.stderr.strip()
    shutil.rmtree(dir_tmp, ignore_errors=True)
    console.print(f"[red]Failed to clone {url}:[/red] {error_msg}")
    raise typer.Exit(1)


def readRepoReadme(path_repo: str, limit_chars: int = 1500) -> str:
    """Read the repo's README for LLM context. Returns "" if not found."""
    matches = list(Path(path_repo).glob("[Rr][Ee][Aa][Dd][Mm][Ee].[Mm][Dd]"))
    if not matches:
        return ""
    try:
        return matches[0].read_text(encoding="utf-8", errors="replace")[:limit_chars]
    except OSError:
        return ""


def executeCommit(message: str) -> None:
    """Run git commit with the given message."""
    result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.print(f"[red]Commit failed:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)
    console.print(f"[green]Committed![/green] {result.stdout.strip()}")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SCORING_RUBRIC = """Score each commit message on a 1-10 scale:
- 1-3: No useful info (wip, fix, update, typo, misc)
- 4-6: Some context but missing scope, type prefix, or details
- 7-8: Good conventional commit with clear scope and description
- 9-10: Excellent - type(scope), detailed body, measurable impact"""


def _buildReadmePrefix(context_repo: str) -> str:
    """Build a prompt prefix from README content. Returns "" if no content."""
    if not context_repo:
        return ""
    return f"Project context (from README):\n{context_repo}\n---\n\n"


def callLlm(prompt: str) -> tuple[str, int]:
    """Call LLM and return (content, total_tokens). Handles auth errors."""
    try:
        response = completion(
            model=resolveModel(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if any(k in error_msg for k in ("auth", "api key", "api_key")):
            console.print(
                "[red]Authentication failed.[/red]\n"
                "Set your API key: export OPENAI_API_KEY=sk-...\n"
                "(or ANTHROPIC_API_KEY for Claude)\n"
                "Choose model: --model gpt-4o-mini (or export LITELLM_MODEL=...)"
            )
            raise typer.Exit(1)
        raise

    content = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


def parseLlmJson(raw: str, model_cls: type[T]) -> T | None:
    """Parse LLM JSON response. Returns None on failure."""
    try:
        data = json.loads(raw)
        return model_cls(**data)
    except Exception:
        return None


def callLlmParsed(prompt: str, model_cls: type[T]) -> tuple[T, int]:
    """Call LLM, parse response, retry once on failure."""
    raw, tokens = callLlm(prompt)
    result = parseLlmJson(raw, model_cls)
    if result is not None:
        return result, tokens

    raw, tokens_retry = callLlm(prompt)
    tokens += tokens_retry
    result = parseLlmJson(raw, model_cls)
    if result is not None:
        return result, tokens

    console.print("[red]Failed to parse LLM response.[/red]")
    console.print(raw[:500], highlight=False, markup=False)
    raise typer.Exit(1)


def _refineSuggestionsWithDiff(
    result: AnalysisResult,
    commits: list[dict],
    repo_path: str,
    context_repo: str = "",
) -> int:
    """Re-analyze worst commits with diff context. Returns tokens used.

    Uses deterministic index mapping -- never trusts LLM-returned identifiers.
    """
    worst_with_idx = [
        (pos, c)
        for pos, c in enumerate(result.commits)
        if c.score < 7
    ]
    if not worst_with_idx:
        return 0

    diff_parts = []
    for idx, (pos, c) in enumerate(worst_with_idx[:5]):
        hash_original = commits[pos]["hash"]
        diff_stat = getCommitDiff(hash_original, repo_path)
        if diff_stat:
            diff_parts.append(f'[{idx}] "{c.message}"\n{diff_stat[:2000]}')

    if not diff_parts:
        return 0

    diff_block = "\n".join(diff_parts)
    prompt = f"""{_buildReadmePrefix(context_repo)}These commits scored poorly. Here's what they actually changed.
Suggest a better message that matches the actual diff.

{diff_block}

Respond with ONLY valid JSON array using the index number.
Each suggestion must be a SINGLE LINE in type(scope): description format, max 72 chars.
[{{"index": 0, "suggestion": "fix(auth): resolve token expiration handling"}}]"""

    raw, tokens = callLlm(prompt)

    try:
        improvements = json.loads(raw)
        if isinstance(improvements, list):
            for item in improvements:
                idx = item["index"]
                if 0 <= idx < len(worst_with_idx):
                    pos_original, _ = worst_with_idx[idx]
                    result.commits[pos_original].suggestion = item["suggestion"]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass  # Phase 2 is best-effort

    return tokens


def analyzeCommits(commits: list[dict], repo_path: str = ".") -> tuple[AnalysisResult, int]:
    """Two-phase diff-aware analysis.

    Phase 1: Batch score all messages (one LLM call).
    Phase 2: Re-analyze worst commits with their diff stat context.
    """
    context_repo = readRepoReadme(repo_path)

    commits_text = "\n".join(
        f"- [{c['hash']}] {c['subject']}"
        + (f"\n  {c['body']}" if c["body"] else "")
        for c in commits
    )

    prompt = f"""{_buildReadmePrefix(context_repo)}Analyze these git commit messages and score each one.

{SCORING_RUBRIC}

For commits scoring < 7, provide:
- "issue": why it's bad (1 sentence)
- "suggestion": a better commit message (single line, type(scope): description format, max 72 chars)

For commits scoring >= 7, provide:
- "praise": why it's good (1 sentence)

Commits:
{commits_text}

Respond with ONLY valid JSON:
{{
    "commits": [
        {{"hash": "abc12345", "message": "the subject", "score": 5, "issue": "...", "suggestion": "fix(auth): resolve token handling", "praise": null}}
    ]
}}"""

    result, tokens_phase1 = callLlmParsed(prompt, AnalysisResult)

    scores = [c.score for c in result.commits]
    result.average_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    result.count_vague = sum(1 for s in scores if s < 7)
    result.count_one_word = sum(
        1 for c in commits if len(c["subject"].split()) == 1
    )

    tokens_phase2 = _refineSuggestionsWithDiff(result, commits, repo_path, context_repo)

    return result, tokens_phase1 + tokens_phase2


def suggestCommitMessage(diff: str, stat: str, repo_path: str = ".") -> tuple[SuggestedCommit, int]:
    """Send staged diff to LLM with smart truncation."""
    context_repo = readRepoReadme(repo_path)

    max_diff_chars = 12000
    diff_truncated = diff[:max_diff_chars]
    if len(diff) > max_diff_chars:
        diff_truncated += f"\n\n... (truncated, {len(diff) - max_diff_chars} chars omitted)"

    prompt = f"""{_buildReadmePrefix(context_repo)}Based on this staged git diff, write a commit message.

File stats:
{stat}

Diff:
{diff_truncated}

Respond with ONLY valid JSON:
{{
    "message": "type(scope): subject\\n\\n- bullet point 1\\n- bullet point 2",
    "changes": ["Modified authentication logic", "Added error handling", "Updated unit tests"]
}}

Rules:
- First line: type(scope): description (under 72 chars)
- Types: feat, fix, refactor, docs, test, chore, perf, ci
- Body: bullet points of specific changes
- Be specific and accurate to the actual diff"""

    return callLlmParsed(prompt, SuggestedCommit)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _formatCommitLines(commit: CommitScore) -> list[str]:
    """Format a single CommitScore into display lines."""
    lines = [
        f'Commit: "{commit.message}"',
        f"Score: {commit.score}/10",
    ]
    if commit.issue:
        lines.append(f"Issue: {commit.issue}")
    if commit.suggestion:
        lines.append(f'Better: "{commit.suggestion}"')
    if commit.praise:
        lines.append(f"Why it's good: {commit.praise}")
    return lines


BLOCK_CHAR = "\u2588"  # Full block for histogram bars

_BUCKET_THRESHOLDS = [3, 6, 8]  # Upper bounds: 1-3, 4-6, 7-8, 9-10
_BUCKET_LABELS = ["1-3", "4-6", "7-8", "9-10"]


def _buildHistogram(commits: list[CommitScore]) -> str:
    """Build a text histogram of score distribution."""
    counts = [0, 0, 0, 0]
    for c in commits:
        counts[bisect_left(_BUCKET_THRESHOLDS, c.score)] += 1

    count_max = max(counts, default=1) or 1
    width_bar = 20
    lines = ["Score distribution:"]
    for label, count in zip(_BUCKET_LABELS, counts):
        bar = BLOCK_CHAR * round(count / count_max * width_bar)
        lines.append(f"  {label:>4}  {bar:<{width_bar}}  {count}")
    return "\n".join(lines)


def _renderCommitPanel(
    commits: list[CommitScore],
    title: str,
    style: str,
    reverse: bool = False,
) -> None:
    """Render a panel of commits sorted by score."""
    if not commits:
        return
    lines: list[str] = []
    for c in sorted(commits, key=lambda x: x.score, reverse=reverse):
        lines.extend(_formatCommitLines(c))
        lines.append("")
    console.print(Panel(
        "\n".join(lines).strip(),
        title=title,
        border_style=style,
    ))


def renderAnalysis(result: AnalysisResult, tokens: int) -> None:
    """Render analysis with rich panels: bad commits, good commits, stats."""
    commits_bad = [c for c in result.commits if c.score < 7]
    commits_good = [c for c in result.commits if c.score >= 7]

    _renderCommitPanel(commits_bad, "COMMITS THAT NEED WORK", "red")
    _renderCommitPanel(commits_good, "WELL-WRITTEN COMMITS", "green", reverse=True)

    count_total = max(len(result.commits), 1)
    stats_lines = [
        f"Average score: {result.average_score}/10",
        f"Vague commits: {result.count_vague} ({result.count_vague * 100 // count_total}%)",
        f"One-word commits: {result.count_one_word} ({result.count_one_word * 100 // count_total}%)",
        "",
        _buildHistogram(result.commits),
        f"Tokens used: {tokens:,}",
    ]
    console.print(Panel("\n".join(stats_lines), title="YOUR STATS", border_style="cyan"))


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    command: str = typer.Argument(None, help="Command: analyze or write"),
    analyze: bool = typer.Option(False, "--analyze", help="Analyze commit message quality."),
    write: bool = typer.Option(False, "--write", help="Suggest a commit message for staged changes."),
    count: int = typer.Option(50, "--count", "-n", help="Number of commits to analyze."),
    url: str | None = typer.Option(None, "--url", "-u", help="Remote repo URL to analyze."),
    branch: str = typer.Option("main", "--branch", "-b", help="Branch to analyze."),
    author: str | None = typer.Option(None, "--author", "-a", help="Filter commits by author name or email."),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM model to use (e.g. gpt-4o, claude-sonnet-4-20250514)."),
) -> None:
    """AI-powered commit message analyzer and writer."""
    global _model
    if model:
        _model = model

    # Support both positional (analyze/write) and flag (--analyze/--write) syntax
    if command == "analyze":
        analyze = True
    elif command == "write":
        write = True
    elif command is not None:
        console.print(f'[red]Unknown command: "{command}"[/red]')
        console.print("Use [bold]--analyze[/bold] or [bold]--write[/bold]. See --help.")
        raise typer.Exit(1)

    if analyze and write:
        console.print("[red]Choose one: --analyze or --write, not both.[/red]")
        raise typer.Exit(1)

    if analyze:
        runAnalyze(count, url, branch, author)
    elif write:
        runWrite()
    else:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def runAnalyze(
    count: int,
    url: str | None,
    branch: str = "main",
    author: str | None = None,
) -> None:
    """Analyze commit message quality in a git repository."""
    dir_repo: str | None = None

    try:
        if url:
            with console.status(f"Cloning {url}..."):
                dir_repo = cloneShallow(url, count)
            path_repo = dir_repo
        else:
            checkGitRepo()
            path_repo = "."

        printStatusLine(path_repo, branch)

        with console.status(f"Reading last {count} commits on [bold]{branch}[/bold]..."):
            commits = getCommits(count, path_repo, branch, author)

        if not commits:
            console.print("[yellow]No commits found.[/yellow]")
            raise typer.Exit(0)

        console.print(f"Analyzing {len(commits)} commits...\n")

        with console.status("Asking the AI to review your commits..."):
            result, tokens = analyzeCommits(commits, path_repo)

        renderAnalysis(result, tokens)

    finally:
        if dir_repo:
            shutil.rmtree(dir_repo, ignore_errors=True)


def runWrite() -> None:
    """Suggest a commit message for your staged changes."""
    checkGitRepo()
    printStatusLine()

    diff = getStagedDiff()
    if not diff.strip():
        console.print("[yellow]No staged changes found. Run 'git add' first.[/yellow]")
        raise typer.Exit(1)

    stat = getStagedStat()
    last_line = stat.split("\n")[-1].strip() if stat else "no stat"
    console.print(f"Analyzing staged changes... ({last_line})\n")

    with console.status("Generating commit message..."):
        suggested, tokens = suggestCommitMessage(diff, stat)

    console.print("Changes detected:")
    for change in suggested.changes:
        console.print(f"  - {change}")
    console.print()
    console.print(Panel(suggested.message, title="Suggested commit message", border_style="green"))
    console.print(f"[dim]Tokens used: {tokens:,}[/dim]\n")

    user_input = Prompt.ask(
        "Press Enter to accept, or type your own message",
        default="",
    )

    message_final = user_input if user_input.strip() else suggested.message
    executeCommit(message_final)


if __name__ == "__main__":
    app()
