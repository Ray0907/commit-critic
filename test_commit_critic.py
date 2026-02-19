"""Tests for commit_critic.py."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit
from rich.console import Console as RichConsole

import commit_critic
from commit_critic import (
    AnalysisResult,
    CommitScore,
    RECORD_SEP,
    SuggestedCommit,
    _buildHistogram,
    _buildReadmePrefix,
    analyzeCommits,
    checkGitRepo,
    getCommits,
    parseLlmJson,
    readRepoReadme,
    renderAnalysis,
    suggestCommitMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _makeLlmResponse(content: str, total_tokens: int = 500) -> MagicMock:
    """Build a mock LLM response with the given JSON content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.total_tokens = total_tokens
    return response


def _makeSubprocessResult(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> MagicMock:
    """Build a mock subprocess.run result."""
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TestModels:
    def test_commit_score_full(self) -> None:
        score = CommitScore(
            hash="abc12345",
            message="fix bug",
            score=2,
            issue="Too vague",
            suggestion="fix(auth): resolve token expiration",
        )
        assert score.score == 2
        assert score.praise is None

    def test_commit_score_good(self) -> None:
        score = CommitScore(
            hash="def67890",
            message="feat(api): add caching",
            score=9,
            praise="Clear scope and type",
        )
        assert score.issue is None
        assert score.suggestion is None

    def test_analysis_result(self) -> None:
        result = AnalysisResult(
            commits=[
                CommitScore(hash="a", message="wip", score=1, issue="No info"),
                CommitScore(hash="b", message="feat: add X", score=8, praise="Good"),
            ],
            average_score=4.5,
            count_vague=1,
            count_one_word=1,
        )
        assert len(result.commits) == 2
        assert result.average_score == 4.5

    def test_suggested_commit(self) -> None:
        suggested = SuggestedCommit(
            message="refactor(auth): extract validation",
            changes=["Extracted validation logic into helper"],
        )
        assert "refactor" in suggested.message
        assert len(suggested.changes) == 1


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

class TestGitOperations:
    def test_get_commits_parses_log(self) -> None:
        output_fake = (
            f"abcdef1234567890{RECORD_SEP}feat: add login{RECORD_SEP}"
            f"body text{RECORD_SEP}\n"
            f"1234567890abcdef{RECORD_SEP}fix bug{RECORD_SEP}{RECORD_SEP}"
        )
        mock_result = _makeSubprocessResult(stdout=output_fake)

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            commits = getCommits(2, ".")

        assert len(commits) == 2
        assert commits[0]["hash"] == "abcdef12"
        assert commits[0]["subject"] == "feat: add login"
        assert commits[0]["body"] == "body text"
        assert commits[1]["subject"] == "fix bug"
        assert commits[1]["body"] == ""

    def test_get_commits_empty_repo(self) -> None:
        mock_result = _makeSubprocessResult(stdout="")

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            commits = getCommits(50, ".")

        assert commits == []

    def test_check_git_repo_fails_outside_repo(self) -> None:
        mock_result = _makeSubprocessResult(
            returncode=128,
            stderr="fatal: not a git repository",
        )

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            with pytest.raises(Exit):
                checkGitRepo("/tmp/not-a-repo")


# ---------------------------------------------------------------------------
# readRepoReadme
# ---------------------------------------------------------------------------

class TestReadRepoReadme:
    def test_reads_readme(self, tmp_path: object) -> None:
        readme = tmp_path / "README.md"  # type: ignore[operator]
        readme.write_text("# My Project\nSome description here.")
        assert readRepoReadme(str(tmp_path)) == "# My Project\nSome description here."

    def test_returns_empty_when_missing(self, tmp_path: object) -> None:
        assert readRepoReadme(str(tmp_path)) == ""

    def test_truncates_to_limit(self, tmp_path: object) -> None:
        readme = tmp_path / "README.md"  # type: ignore[operator]
        readme.write_text("x" * 3000)
        result = readRepoReadme(str(tmp_path), limit_chars=100)
        assert len(result) == 100

    def test_case_insensitive(self, tmp_path: object) -> None:
        readme = tmp_path / "readme.md"  # type: ignore[operator]
        readme.write_text("lowercase readme")
        assert readRepoReadme(str(tmp_path)) == "lowercase readme"


class TestBuildReadmePrefix:
    def test_returns_empty_for_empty_input(self) -> None:
        assert _buildReadmePrefix("") == ""

    def test_wraps_content_with_header_and_separator(self) -> None:
        result = _buildReadmePrefix("# My Project")
        assert result.startswith("Project context (from README):")
        assert "# My Project" in result
        assert result.endswith("---\n\n")


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

class TestLlmHelpers:
    @pytest.fixture(autouse=True)
    def set_model(self) -> None:
        commit_critic._model = "test-model"
        yield
        commit_critic._model = None

    def test_parse_llm_json_valid(self) -> None:
        raw = json.dumps({
            "commits": [
                {
                    "hash": "abc",
                    "message": "wip",
                    "score": 1,
                    "issue": "No info",
                    "suggestion": "Describe changes",
                    "praise": None,
                }
            ],
        })
        result = parseLlmJson(raw, AnalysisResult)
        assert result is not None
        assert result.commits[0].score == 1

    def test_parse_llm_json_invalid(self) -> None:
        result = parseLlmJson("not json {{{", AnalysisResult)
        assert result is None

    def test_parse_llm_json_wrong_schema(self) -> None:
        result = parseLlmJson('{"wrong": "schema"}', AnalysisResult)
        assert result is None

    def test_analyze_commits_calls_llm(self) -> None:
        response_fake = _makeLlmResponse(
            content=json.dumps({
                "commits": [
                    {
                        "hash": "abc12345",
                        "message": "wip",
                        "score": 2,
                        "issue": "Vague",
                        "suggestion": "Be specific",
                        "praise": None,
                    },
                    {
                        "hash": "def67890",
                        "message": "feat: add X",
                        "score": 8,
                        "issue": None,
                        "suggestion": None,
                        "praise": "Clear",
                    },
                ],
            }),
            total_tokens=500,
        )

        with patch("commit_critic.completion", return_value=response_fake), \
             patch("commit_critic.readRepoReadme", return_value=""):
            result, tokens = analyzeCommits([
                {"hash": "abc12345", "subject": "wip", "body": ""},
                {"hash": "def67890", "subject": "feat: add X", "body": ""},
            ])

        assert len(result.commits) == 2
        assert result.average_score == 5.0  # (2+8)/2
        assert result.count_vague == 1      # score < 7: "wip" (score=2)
        assert result.count_one_word == 1   # "wip" is one word
        assert tokens == 500

    def test_phase2_uses_deterministic_index(self) -> None:
        """Phase 2 should use index-based mapping, not LLM-returned hashes."""
        response_phase1 = _makeLlmResponse(
            content=json.dumps({
                "commits": [
                    {
                        "hash": "aaa",
                        "message": "fix",
                        "score": 2,
                        "issue": "Vague",
                        "suggestion": "old suggestion",
                        "praise": None,
                    },
                    {
                        "hash": "bbb",
                        "message": "feat: good",
                        "score": 8,
                        "issue": None,
                        "suggestion": None,
                        "praise": "Nice",
                    },
                ],
            }),
            total_tokens=400,
        )

        response_phase2 = _makeLlmResponse(
            content=json.dumps([
                {"index": 0, "suggestion": "fix(auth): resolve token expiration"}
            ]),
            total_tokens=200,
        )

        with patch("commit_critic.completion", side_effect=[response_phase1, response_phase2]), \
             patch("commit_critic.getCommitDiff", return_value="file.py | 10 +++---"), \
             patch("commit_critic.readRepoReadme", return_value=""):
            result, tokens = analyzeCommits([
                {"hash": "aaaa1234", "subject": "fix", "body": ""},
                {"hash": "bbbb5678", "subject": "feat: good", "body": ""},
            ])

        assert result.commits[0].suggestion == "fix(auth): resolve token expiration"
        assert tokens == 600

    def test_suggest_commit_message(self) -> None:
        response_fake = _makeLlmResponse(
            content=json.dumps({
                "message": "feat(auth): add login endpoint",
                "changes": ["Added login endpoint", "Configured JWT tokens"],
            }),
            total_tokens=300,
        )

        with patch("commit_critic.completion", return_value=response_fake), \
             patch("commit_critic.readRepoReadme", return_value=""):
            result, tokens = suggestCommitMessage("diff content", "1 file changed")

        assert "feat" in result.message
        assert tokens == 300

    def test_suggest_commit_truncates_large_diff(self) -> None:
        diff_large = "x" * 20000
        response_fake = _makeLlmResponse(
            content=json.dumps({
                "message": "chore: update",
                "changes": ["Updated files"],
            }),
            total_tokens=200,
        )

        with patch("commit_critic.completion", return_value=response_fake) as mock_comp, \
             patch("commit_critic.readRepoReadme", return_value=""):
            suggestCommitMessage(diff_large, "stat")
            prompt_sent = mock_comp.call_args[1]["messages"][0]["content"]
            assert "truncated" in prompt_sent
            assert len(prompt_sent) < 20000

    def test_suggest_commit_includes_readme_context(self) -> None:
        response_fake = _makeLlmResponse(
            content=json.dumps({
                "message": "feat(api): add endpoint",
                "changes": ["Added endpoint"],
            }),
            total_tokens=200,
        )

        with patch("commit_critic.completion", return_value=response_fake) as mock_comp, \
             patch("commit_critic.readRepoReadme", return_value="# Cool Project"):
            suggestCommitMessage("diff", "stat")
            prompt_sent = mock_comp.call_args[1]["messages"][0]["content"]
            assert "Project context (from README):" in prompt_sent
            assert "# Cool Project" in prompt_sent


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestOutputFormatting:
    @pytest.fixture()
    def capture_console(self) -> StringIO:
        """Temporarily replace commit_critic.console to capture output."""
        output = StringIO()
        console_test = RichConsole(file=output, force_terminal=True)
        original = commit_critic.console
        commit_critic.console = console_test
        yield output
        commit_critic.console = original

    def test_render_analysis_shows_bad_and_good(self, capture_console: StringIO) -> None:
        result = AnalysisResult(
            commits=[
                CommitScore(hash="a", message="wip", score=1, issue="No info", suggestion="Be specific"),
                CommitScore(hash="b", message="feat: add login", score=9, praise="Clear and scoped"),
            ],
            average_score=5.0,
            count_vague=1,
            count_one_word=1,
        )

        renderAnalysis(result, 500)

        rendered = capture_console.getvalue()
        assert "wip" in rendered
        assert "feat: add login" in rendered
        assert "5.0" in rendered

    def test_render_analysis_no_division_by_zero(self, capture_console: StringIO) -> None:
        result = AnalysisResult(
            commits=[],
            average_score=0.0,
            count_vague=0,
            count_one_word=0,
        )

        renderAnalysis(result, 0)

    def test_render_analysis_includes_histogram(self, capture_console: StringIO) -> None:
        result = AnalysisResult(
            commits=[
                CommitScore(hash="a", message="wip", score=1, issue="Bad"),
                CommitScore(hash="b", message="fix", score=3, issue="Vague"),
                CommitScore(hash="c", message="feat: add X", score=8, praise="Good"),
            ],
            average_score=4.0,
            count_vague=2,
            count_one_word=2,
        )

        renderAnalysis(result, 100)

        rendered = capture_console.getvalue()
        assert "Score distribution" in rendered
        assert "1-3" in rendered
        assert "7-8" in rendered


class TestHistogram:
    def test_histogram_buckets(self) -> None:
        commits = [
            CommitScore(hash="a", message="wip", score=1, issue="Bad"),
            CommitScore(hash="b", message="fix", score=2, issue="Bad"),
            CommitScore(hash="c", message="update", score=5, issue="Meh"),
            CommitScore(hash="d", message="feat: X", score=9, praise="Great"),
        ]
        output = _buildHistogram(commits)
        assert "1-3" in output
        assert "4-6" in output
        assert "9-10" in output
        # Verify actual counts: 2 in 1-3, 1 in 4-6, 0 in 7-8, 1 in 9-10
        lines = output.strip().split("\n")
        count_line_13 = [l for l in lines if "1-3" in l][0]
        count_line_46 = [l for l in lines if "4-6" in l][0]
        count_line_78 = [l for l in lines if "7-8" in l][0]
        assert count_line_13.strip().endswith("2")
        assert count_line_46.strip().endswith("1")
        assert count_line_78.strip().endswith("0")

    def test_histogram_empty(self) -> None:
        output = _buildHistogram([])
        assert "Score distribution" in output


class TestAuthorFilter:
    def test_get_commits_passes_author_flag(self) -> None:
        mock_result = _makeSubprocessResult(stdout="")

        with patch("commit_critic.subprocess.run", return_value=mock_result) as mock_run:
            getCommits(10, ".", author="ray")

        cmd = mock_run.call_args[0][0]
        assert "--author" in cmd
        assert "ray" in cmd

    def test_get_commits_no_author(self) -> None:
        mock_result = _makeSubprocessResult(stdout="")

        with patch("commit_critic.subprocess.run", return_value=mock_result) as mock_run:
            getCommits(10, ".")

        cmd = mock_run.call_args[0][0]
        assert "--author" not in cmd
