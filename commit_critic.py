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


if __name__ == "__main__":
    app()
