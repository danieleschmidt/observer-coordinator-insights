import json
from typing import Dict, List
from .profile import PersonProfile
from .team import TeamComposer
from .cluster import ClusterAnalyzer


def generate_team_report(profiles: List[PersonProfile], k_clusters: int = None) -> Dict:
    """Generate a full team analytics report."""
    k = k_clusters or max(2, len(profiles) // 3)

    composer = TeamComposer(profiles)
    summary = composer.team_summary()
    hire_rec = composer.recommend_hire() if profiles else {}

    cluster_summary = {}
    if len(profiles) >= 2:
        analyzer = ClusterAnalyzer(k=k)
        analyzer.fit(profiles)
        cluster_summary = analyzer.cluster_summary()

    return {
        "team_summary": summary,
        "hire_recommendation": hire_rec,
        "clusters": cluster_summary,
        "insights": _generate_insights(composer),
    }


def _generate_insights(composer: TeamComposer) -> List[str]:
    insights = []
    dist = composer.color_distribution

    if dist["red"] > dist["green"]:
        insights.append("Team leans competitive — ensure space for harmonizing voices.")
    if dist["blue"] > dist["yellow"]:
        insights.append("Strong analytical presence — may need to consciously foster creativity.")
    if dist["yellow"] > dist["blue"]:
        insights.append("Energetic and social team — consider structure to channel enthusiasm.")
    if composer.color_balance_score > 0.8:
        insights.append("Team is well-balanced across all color energies.")
    if not insights:
        insights.append("Review distribution for potential blind spots.")

    return insights


def report_to_json(profiles: List[PersonProfile], **kwargs) -> str:
    report = generate_team_report(profiles, **kwargs)
    return json.dumps(report, indent=2)
