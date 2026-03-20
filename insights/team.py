from typing import List, Dict, Optional
from .profile import PersonProfile, QUADRANTS


class TeamComposer:
    """Analyzes team composition and recommends balanced additions."""

    def __init__(self, profiles: List[PersonProfile]):
        self.profiles = profiles

    @property
    def color_distribution(self) -> Dict[str, int]:
        """Count of dominant colors in team."""
        counts = {c: 0 for c in QUADRANTS}
        for p in self.profiles:
            counts[p.dominant_color] += 1
        return counts

    @property
    def color_balance_score(self) -> float:
        """1.0 = perfectly balanced, 0.0 = all one color."""
        if not self.profiles:
            return 0.0
        dist = self.color_distribution
        total = sum(dist.values())
        expected = total / len(QUADRANTS)
        variance = sum((v - expected) ** 2 for v in dist.values()) / len(QUADRANTS)
        max_variance = expected ** 2  # worst case: all in one
        return max(0.0, 1.0 - variance / max_variance) if max_variance > 0 else 1.0

    @property
    def missing_colors(self) -> List[str]:
        """Colors not represented in team."""
        dist = self.color_distribution
        return [c for c, count in dist.items() if count == 0]

    @property
    def overrepresented_colors(self) -> List[str]:
        """Colors with more than expected members."""
        if not self.profiles:
            return []
        dist = self.color_distribution
        expected = len(self.profiles) / len(QUADRANTS)
        return [c for c, count in dist.items() if count > expected * 1.5]

    def recommend_hire(self) -> Dict:
        """Recommend what color profile to hire next."""
        missing = self.missing_colors
        over = self.overrepresented_colors

        if missing:
            priority = missing[0]
            reason = f"No {priority} profiles on team — adds missing energy"
        elif over:
            # Hire for least represented
            dist = self.color_distribution
            priority = min(QUADRANTS, key=lambda c: dist[c])
            reason = f"{over[0]} is overrepresented — balance with {priority}"
        else:
            dist = self.color_distribution
            priority = min(QUADRANTS, key=lambda c: dist[c])
            reason = f"Slight {priority} underrepresentation"

        from .profile import QUADRANT_TRAITS
        return {
            "recommended_color": priority,
            "reason": reason,
            "traits_to_look_for": QUADRANT_TRAITS[priority],
        }

    def team_summary(self) -> Dict:
        return {
            "size": len(self.profiles),
            "color_distribution": self.color_distribution,
            "balance_score": round(self.color_balance_score, 3),
            "missing_colors": self.missing_colors,
            "overrepresented": self.overrepresented_colors,
            "members": [
                {"name": p.name, "dominant": p.dominant_color, "scores": p.scores}
                for p in self.profiles
            ],
        }
