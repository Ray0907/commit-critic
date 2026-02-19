"""Tests for commit_critic.py."""

import json
import pytest
from unittest.mock import patch, MagicMock
from commit_critic import CommitScore, AnalysisResult, SuggestedCommit


class TestModels:
    def test_commit_score_full(self):
        score = CommitScore(
            hash="abc12345",
            message="fix bug",
            score=2,
            issue="Too vague",
            suggestion="fix(auth): resolve token expiration",
        )
        assert score.score == 2
        assert score.praise is None

    def test_commit_score_good(self):
        score = CommitScore(
            hash="def67890",
            message="feat(api): add caching",
            score=9,
            praise="Clear scope and type",
        )
        assert score.issue is None
        assert score.suggestion is None

    def test_analysis_result(self):
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

    def test_suggested_commit(self):
        s = SuggestedCommit(
            message="refactor(auth): extract validation",
            summary="Extracted validation logic into helper",
        )
        assert "refactor" in s.message


class TestGitOperations:
    def test_get_commits_parses_log(self):
        from commit_critic import getCommits, RECORD_SEP

        fake_output = (
            f"abcdef1234567890{RECORD_SEP}feat: add login{RECORD_SEP}body text{RECORD_SEP}\n"
            f"1234567890abcdef{RECORD_SEP}fix bug{RECORD_SEP}{RECORD_SEP}"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            commits = getCommits(2, ".")

        assert len(commits) == 2
        assert commits[0]["hash"] == "abcdef12"
        assert commits[0]["subject"] == "feat: add login"
        assert commits[0]["body"] == "body text"
        assert commits[1]["subject"] == "fix bug"
        assert commits[1]["body"] == ""

    def test_get_commits_empty_repo(self):
        from commit_critic import getCommits

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            commits = getCommits(50, ".")

        assert commits == []

    def test_check_git_repo_fails_outside_repo(self):
        from commit_critic import checkGitRepo

        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stderr = "fatal: not a git repository"

        from click.exceptions import Exit

        with patch("commit_critic.subprocess.run", return_value=mock_result):
            with pytest.raises(Exit):
                checkGitRepo("/tmp/not-a-repo")
