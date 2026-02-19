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


class TestLlmHelpers:
    def test_parse_llm_json_valid(self):
        from commit_critic import parseLlmJson

        raw = json.dumps({
            "commits": [
                {"hash": "abc", "message": "wip", "score": 1, "issue": "No info", "suggestion": "Describe changes", "praise": None}
            ],
            "average_score": 1.0,
            "count_vague": 1,
            "count_one_word": 1,
        })
        result = parseLlmJson(raw, AnalysisResult)
        assert result is not None
        assert result.commits[0].score == 1

    def test_parse_llm_json_invalid(self):
        from commit_critic import parseLlmJson

        result = parseLlmJson("not json {{{", AnalysisResult)
        assert result is None

    def test_parse_llm_json_wrong_schema(self):
        from commit_critic import parseLlmJson

        result = parseLlmJson('{"wrong": "schema"}', AnalysisResult)
        assert result is None

    def test_analyze_commits_calls_llm(self):
        from commit_critic import analyzeCommits

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = json.dumps({
            "commits": [
                {"hash": "abc12345", "message": "wip", "score": 2, "issue": "Vague", "suggestion": "Be specific", "praise": None},
                {"hash": "def67890", "message": "feat: add X", "score": 8, "issue": None, "suggestion": None, "praise": "Clear"},
            ],
            "average_score": 5.0,
            "count_vague": 1,
            "count_one_word": 1,
        })
        fake_response.usage = MagicMock()
        fake_response.usage.total_tokens = 500

        with patch("commit_critic.completion", return_value=fake_response):
            result, tokens = analyzeCommits([
                {"hash": "abc12345", "subject": "wip", "body": ""},
                {"hash": "def67890", "subject": "feat: add X", "body": ""},
            ])

        assert len(result.commits) == 2
        assert result.average_score == 5.0
        assert tokens == 500

    def test_phase2_uses_deterministic_index(self):
        """Phase 2 should use index-based mapping, not LLM-returned hashes."""
        from commit_critic import analyzeCommits

        phase1_response = MagicMock()
        phase1_response.choices = [MagicMock()]
        phase1_response.choices[0].message.content = json.dumps({
            "commits": [
                {"hash": "aaa", "message": "fix", "score": 2, "issue": "Vague", "suggestion": "old suggestion", "praise": None},
                {"hash": "bbb", "message": "feat: good", "score": 8, "issue": None, "suggestion": None, "praise": "Nice"},
            ],
            "average_score": 5.0,
            "count_vague": 1,
            "count_one_word": 1,
        })
        phase1_response.usage = MagicMock()
        phase1_response.usage.total_tokens = 400

        phase2_response = MagicMock()
        phase2_response.choices = [MagicMock()]
        phase2_response.choices[0].message.content = json.dumps([
            {"index": 0, "suggestion": "fix(auth): resolve token expiration"}
        ])
        phase2_response.usage = MagicMock()
        phase2_response.usage.total_tokens = 200

        with patch("commit_critic.completion", side_effect=[phase1_response, phase2_response]), \
             patch("commit_critic.getCommitDiff", return_value="file.py | 10 +++---"):
            result, tokens = analyzeCommits([
                {"hash": "aaaa1234", "subject": "fix", "body": ""},
                {"hash": "bbbb5678", "subject": "feat: good", "body": ""},
            ])

        assert result.commits[0].suggestion == "fix(auth): resolve token expiration"
        assert tokens == 600

    def test_suggest_commit_message(self):
        from commit_critic import suggestCommitMessage

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = json.dumps({
            "message": "feat(auth): add login endpoint",
            "summary": "Added login endpoint with JWT tokens",
        })
        fake_response.usage = MagicMock()
        fake_response.usage.total_tokens = 300

        with patch("commit_critic.completion", return_value=fake_response):
            result, tokens = suggestCommitMessage("diff content", "1 file changed")

        assert "feat" in result.message
        assert tokens == 300

    def test_suggest_commit_truncates_large_diff(self):
        from commit_critic import suggestCommitMessage

        large_diff = "x" * 20000
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = json.dumps({
            "message": "chore: update",
            "summary": "Updated files",
        })
        fake_response.usage = MagicMock()
        fake_response.usage.total_tokens = 200

        with patch("commit_critic.completion", return_value=fake_response) as mock_comp:
            suggestCommitMessage(large_diff, "stat")
            prompt_sent = mock_comp.call_args[1]["messages"][0]["content"]
            assert "truncated" in prompt_sent
            assert len(prompt_sent) < 20000
