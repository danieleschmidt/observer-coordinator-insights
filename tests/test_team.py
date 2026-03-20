import pytest
from insights.profile import PersonProfile
from insights.team import TeamComposer


def make_team():
    return [
        PersonProfile("Alice", red=80, yellow=10, green=5, blue=5),
        PersonProfile("Bob", red=75, yellow=15, green=5, blue=5),
        PersonProfile("Carol", red=5, yellow=80, green=10, blue=5),
    ]


def test_color_distribution():
    team = make_team()
    tc = TeamComposer(team)
    dist = tc.color_distribution
    assert dist["red"] == 2
    assert dist["yellow"] == 1


def test_missing_colors():
    team = make_team()
    tc = TeamComposer(team)
    missing = tc.missing_colors
    assert "green" in missing
    assert "blue" in missing


def test_balance_score_low():
    team = make_team()  # skewed red
    tc = TeamComposer(team)
    assert tc.color_balance_score < 0.9


def test_balance_score_high():
    team = [
        PersonProfile("A", red=80),
        PersonProfile("B", yellow=80),
        PersonProfile("C", green=80),
        PersonProfile("D", blue=80),
    ]
    tc = TeamComposer(team)
    assert tc.color_balance_score > 0.8


def test_recommend_hire():
    team = make_team()
    tc = TeamComposer(team)
    rec = tc.recommend_hire()
    assert "recommended_color" in rec
    assert "reason" in rec
    assert "traits_to_look_for" in rec


def test_team_summary():
    team = make_team()
    tc = TeamComposer(team)
    summary = tc.team_summary()
    assert summary["size"] == 3
    assert "color_distribution" in summary
    assert "balance_score" in summary


def test_empty_team():
    tc = TeamComposer([])
    assert tc.color_balance_score == 0.0
    assert tc.missing_colors == ["red", "yellow", "green", "blue"]
