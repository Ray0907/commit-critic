"""Tests for commit_critic.py."""

import json
import pytest
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
