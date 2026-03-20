from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import math

QUADRANTS = ["red", "yellow", "green", "blue"]

QUADRANT_TRAITS = {
    "red": ["competitive", "decisive", "demanding", "determined", "strong-willed"],
    "yellow": ["enthusiastic", "sociable", "dynamic", "demonstrative", "creative"],
    "green": ["caring", "patient", "harmonious", "sharing", "reliable"],
    "blue": ["analytical", "precise", "cautious", "deliberate", "questioning"],
}


@dataclass
class PersonProfile:
    name: str
    red: float = 0.0    # score 0-100
    yellow: float = 0.0
    green: float = 0.0
    blue: float = 0.0

    def __post_init__(self):
        for q in QUADRANTS:
            val = getattr(self, q)
            if not 0 <= val <= 100:
                raise ValueError(f"{q} score must be 0-100, got {val}")

    @property
    def dominant_color(self) -> str:
        """Return the quadrant with the highest score."""
        return max(QUADRANTS, key=lambda q: getattr(self, q))

    @property
    def scores(self) -> Dict[str, float]:
        return {q: getattr(self, q) for q in QUADRANTS}

    @property
    def vector(self) -> List[float]:
        """Return scores as a list for clustering."""
        return [self.red, self.yellow, self.green, self.blue]

    @property
    def total_score(self) -> float:
        return sum(self.vector)

    @property
    def normalized_vector(self) -> List[float]:
        total = self.total_score or 1.0
        return [v / total for v in self.vector]

    def traits(self) -> List[str]:
        return QUADRANT_TRAITS.get(self.dominant_color, [])

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersonProfile":
        return cls(
            name=d["name"],
            red=d.get("red", 0),
            yellow=d.get("yellow", 0),
            green=d.get("green", 0),
            blue=d.get("blue", 0),
        )
