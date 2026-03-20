import pytest
from insights.profile import PersonProfile
from insights.cluster import ClusterAnalyzer


def make_profiles():
    return [
        PersonProfile("Alice", red=80, yellow=10, green=10, blue=10),
        PersonProfile("Bob", red=70, yellow=20, green=5, blue=5),
        PersonProfile("Carol", red=5, yellow=5, green=80, blue=20),
        PersonProfile("Dave", red=5, yellow=5, green=70, blue=30),
        PersonProfile("Eve", red=10, yellow=10, green=10, blue=80),
    ]


def test_cluster_fit_runs():
    profiles = make_profiles()
    ca = ClusterAnalyzer(k=2)
    ca.fit(profiles)
    assert len(ca.labels) == len(profiles)


def test_cluster_labels_count():
    profiles = make_profiles()
    ca = ClusterAnalyzer(k=3)
    ca.fit(profiles)
    assert len(ca.clusters) <= 3


def test_cluster_summary():
    profiles = make_profiles()
    ca = ClusterAnalyzer(k=2)
    ca.fit(profiles)
    summary = ca.cluster_summary()
    assert isinstance(summary, dict)
    total = sum(s["size"] for s in summary.values())
    assert total == len(profiles)


def test_predict():
    profiles = make_profiles()
    ca = ClusterAnalyzer(k=2)
    ca.fit(profiles)
    new = PersonProfile("New", red=75, yellow=10, green=5, blue=10)
    label = ca.predict(new)
    assert label in ca.clusters


def test_cluster_k_larger_than_profiles():
    profiles = [PersonProfile("A", red=80), PersonProfile("B", blue=80)]
    ca = ClusterAnalyzer(k=10)
    ca.fit(profiles)
    assert len(ca.clusters) <= len(profiles)


def test_cluster_empty_raises():
    ca = ClusterAnalyzer(k=2)
    with pytest.raises(ValueError):
        ca.fit([])
