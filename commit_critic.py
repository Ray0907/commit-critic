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

app = typer.Typer(help="AI-powered commit message analyzer and writer.")
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


if __name__ == "__main__":
    app()
