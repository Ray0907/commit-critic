"""AI Commit Message Critic - Analyze and improve your commit messages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import typer
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()

app = typer.Typer(help="AI-powered commit message analyzer and writer.", invoke_without_command=True)
console = Console()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def getModel() -> str:
    """Get the LLM model from env or default."""
    return os.environ.get("LITELLM_MODEL", "gpt-4o-mini")


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
    average_score: float
    count_vague: int
    count_one_word: int


class SuggestedCommit(BaseModel):
    message: str
    summary: str


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

RECORD_SEP = "\x1e"  # ASCII record separator (from pre-commit pattern)


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


def getCommits(count: int, repo_path: str = ".") -> list[dict]:
    """Extract commits from git log. Returns list of {hash, subject, body}."""
    fmt = f"%H{RECORD_SEP}%s{RECORD_SEP}%b{RECORD_SEP}"
    result = subprocess.run(
        ["git", "-C", repo_path, "log", f"--format={fmt}", "-n", str(count)],
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


def cloneShallow(url: str, count: int) -> str:
    """Shallow clone a repo to a temp directory. Returns the path."""
    dir_tmp = tempfile.mkdtemp(prefix="commit_critic_")
    result = subprocess.run(
        ["git", "clone", "--depth", str(count), url, dir_tmp],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(dir_tmp, ignore_errors=True)
        console.print(f"[red]Failed to clone {url}:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)
    return dir_tmp


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


def callLlm(prompt: str) -> tuple[str, int]:
    """Call LLM and return (content, total_tokens). Handles auth errors."""
    try:
        response = completion(
            model=getModel(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "auth" in error_msg or "api key" in error_msg or "api_key" in error_msg:
            console.print(
                "[red]Authentication failed.[/red]\n"
                "Set your API key: export OPENAI_API_KEY=sk-...\n"
                "(or ANTHROPIC_API_KEY for Claude)\n"
                "Choose model: export LITELLM_MODEL=gpt-4o-mini"
            )
            raise typer.Exit(1)
        raise

    content = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


def parseLlmJson(raw: str, model_cls: type):
    """Parse LLM JSON response. Returns None on failure."""
    try:
        data = json.loads(raw)
        return model_cls(**data)
    except Exception:
        return None


def analyzeCommits(commits: list[dict], repo_path: str = ".") -> tuple[AnalysisResult, int]:
    """
    Two-phase diff-aware analysis:
    1. Batch score all messages (one LLM call)
    2. Re-analyze worst commits with their diff stat context
    Returns (result, total_tokens).
    """
    total_tokens = 0

    # Phase 1: Batch score
    commits_text = "\n".join(
        f"- [{c['hash']}] {c['subject']}"
        + (f"\n  {c['body']}" if c['body'] else "")
        for c in commits
    )

    prompt_phase1 = f"""Analyze these git commit messages and score each one.

{SCORING_RUBRIC}

For commits scoring < 7, provide:
- "issue": why it's bad (1 sentence)
- "suggestion": a better commit message

For commits scoring >= 7, provide:
- "praise": why it's good (1 sentence)

Also calculate:
- average_score: mean of all scores (1 decimal)
- count_vague: commits with score <= 5
- count_one_word: commits where the subject is a single word

Commits:
{commits_text}

Respond with ONLY valid JSON:
{{
    "commits": [
        {{"hash": "abc12345", "message": "the subject", "score": 5, "issue": "...", "suggestion": "...", "praise": null}}
    ],
    "average_score": 5.0,
    "count_vague": 10,
    "count_one_word": 3
}}"""

    raw, tokens = callLlm(prompt_phase1)
    total_tokens += tokens

    result = parseLlmJson(raw, AnalysisResult)
    if result is None:
        raw, tokens = callLlm(prompt_phase1)
        total_tokens += tokens
        result = parseLlmJson(raw, AnalysisResult)
        if result is None:
            console.print("[red]Failed to parse LLM response.[/red]")
            console.print(raw[:500])
            raise typer.Exit(1)

    # Phase 2: Diff-aware re-analysis for worst commits
    # Deterministic index mapping - never trust LLM-returned identifiers
    worst_with_idx = [
        (i, c) for i, c in enumerate(result.commits) if c.score <= 4
    ]
    if worst_with_idx:
        diff_parts = []
        for idx, (i, c) in enumerate(worst_with_idx[:5]):
            original_hash = commits[i]["hash"]
            diff_stat = getCommitDiff(original_hash, repo_path)
            if diff_stat:
                diff_stat = diff_stat[:2000]
                diff_parts.append(f"[{idx}] \"{c.message}\"\n{diff_stat}")

        if diff_parts:
            prompt_phase2 = f"""These commits scored poorly. Here's what they actually changed.
Suggest a better message that matches the actual diff.

{chr(10).join(diff_parts)}

Respond with ONLY valid JSON array using the index number:
[{{"index": 0, "suggestion": "better message based on actual changes"}}]"""

            raw2, tokens2 = callLlm(prompt_phase2)
            total_tokens += tokens2

            try:
                improvements = json.loads(raw2)
                if isinstance(improvements, list):
                    for item in improvements:
                        idx = item["index"]
                        if 0 <= idx < len(worst_with_idx):
                            orig_idx, _ = worst_with_idx[idx]
                            result.commits[orig_idx].suggestion = item["suggestion"]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass  # Phase 2 is best-effort

    return result, total_tokens


def suggestCommitMessage(diff: str, stat: str) -> tuple[SuggestedCommit, int]:
    """Send staged diff to LLM with smart truncation."""
    max_diff_chars = 12000
    diff_truncated = diff[:max_diff_chars]
    if len(diff) > max_diff_chars:
        diff_truncated += f"\n\n... (truncated, {len(diff) - max_diff_chars} chars omitted)"

    prompt = f"""Based on this staged git diff, write a commit message.

File stats:
{stat}

Diff:
{diff_truncated}

Respond with ONLY valid JSON:
{{
    "message": "type(scope): subject\\n\\n- bullet point 1\\n- bullet point 2",
    "summary": "Human readable summary of what changed"
}}

Rules:
- First line: type(scope): description (under 72 chars)
- Types: feat, fix, refactor, docs, test, chore, perf, ci
- Body: bullet points of specific changes
- Be specific and accurate to the actual diff"""

    raw, tokens = callLlm(prompt)
    result = parseLlmJson(raw, SuggestedCommit)
    if result is None:
        raw, tokens2 = callLlm(prompt)
        tokens += tokens2
        result = parseLlmJson(raw, SuggestedCommit)
        if result is None:
            console.print("[red]Failed to parse LLM response.[/red]")
            console.print(raw[:500])
            raise typer.Exit(1)

    return result, tokens


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def renderAnalysis(result: AnalysisResult, tokens: int) -> None:
    """Render analysis with rich panels: bad commits, good commits, stats."""
    bad_commits = [c for c in result.commits if c.score < 7]
    good_commits = [c for c in result.commits if c.score >= 7]

    if bad_commits:
        lines = []
        for c in sorted(bad_commits, key=lambda x: x.score):
            lines.append(f'Commit: "{c.message}"')
            lines.append(f"Score: {c.score}/10")
            if c.issue:
                lines.append(f"Issue: {c.issue}")
            if c.suggestion:
                lines.append(f'Better: "{c.suggestion}"')
            lines.append("")
        console.print(Panel(
            "\n".join(lines).strip(),
            title="COMMITS THAT NEED WORK",
            border_style="red",
        ))

    if good_commits:
        lines = []
        for c in sorted(good_commits, key=lambda x: x.score, reverse=True):
            lines.append(f'Commit: "{c.message}"')
            lines.append(f"Score: {c.score}/10")
            if c.praise:
                lines.append(f"Why it's good: {c.praise}")
            lines.append("")
        console.print(Panel(
            "\n".join(lines).strip(),
            title="WELL-WRITTEN COMMITS",
            border_style="green",
        ))

    total = max(len(result.commits), 1)
    stats_lines = [
        f"Average score: {result.average_score}/10",
        f"Vague commits: {result.count_vague} ({result.count_vague * 100 // total}%)",
        f"One-word commits: {result.count_one_word} ({result.count_one_word * 100 // total}%)",
        f"Tokens used: {tokens:,}",
    ]
    console.print(Panel("\n".join(stats_lines), title="YOUR STATS", border_style="cyan"))


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    ctx: typer.Context,
    do_analyze: bool = typer.Option(False, "--analyze", help="Analyze commit message quality."),
    do_write: bool = typer.Option(False, "--write", help="Suggest a commit message for staged changes."),
    count: int = typer.Option(50, "--count", "-n", help="Number of commits to analyze."),
    url: str | None = typer.Option(None, "--url", "-u", help="Remote repo URL to analyze."),
) -> None:
    """AI-powered commit message analyzer and writer."""
    if not do_analyze and not do_write:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    if do_analyze and do_write:
        console.print("[red]Choose one: --analyze or --write, not both.[/red]")
        raise typer.Exit(1)

    if do_analyze:
        runAnalyze(count, url)
    else:
        runWrite()


def runAnalyze(count: int, url: str | None) -> None:
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

        with console.status(f"Reading last {count} commits..."):
            commits = getCommits(count, path_repo)

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

    diff = getStagedDiff()
    if not diff.strip():
        console.print("[yellow]No staged changes found. Run 'git add' first.[/yellow]")
        raise typer.Exit(1)

    stat = getStagedStat()
    last_line = stat.split("\n")[-1].strip() if stat else "no stat"
    console.print(f"Analyzing staged changes... ({last_line})\n")

    with console.status("Generating commit message..."):
        suggested, tokens = suggestCommitMessage(diff, stat)

    console.print(f"Changes detected:\n{suggested.summary}\n")
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
