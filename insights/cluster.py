import math
import random
from typing import List, Dict, Tuple
from .profile import PersonProfile, QUADRANTS


def _distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return [0.0] * 4
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(4)]


class ClusterAnalyzer:
    """Groups PersonProfiles into k clusters using k-means."""

    def __init__(self, k: int = 3, max_iter: int = 100, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.clusters: Dict[int, List[PersonProfile]] = {}

    def fit(self, profiles: List[PersonProfile]) -> "ClusterAnalyzer":
        if not profiles:
            raise ValueError("Need at least one profile to cluster")
        k = min(self.k, len(profiles))

        random.seed(self.seed)
        vectors = [p.vector for p in profiles]

        # Initialize centroids randomly
        indices = random.sample(range(len(vectors)), k)
        centroids = [vectors[i][:] for i in indices]

        labels = [0] * len(vectors)

        for _ in range(self.max_iter):
            # Assignment
            new_labels = []
            for vec in vectors:
                dists = [_distance(vec, c) for c in centroids]
                new_labels.append(dists.index(min(dists)))

            if new_labels == labels:
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                members = [vectors[i] for i, l in enumerate(labels) if l == j]
                if members:
                    centroids[j] = _centroid(members)

        self.centroids = centroids
        self.labels = labels
        self.clusters = {}
        for i, (label, profile) in enumerate(zip(labels, profiles)):
            self.clusters.setdefault(label, []).append(profile)

        return self

    def cluster_summary(self) -> Dict[int, Dict]:
        """Return summary stats per cluster."""
        summary = {}
        for cluster_id, members in self.clusters.items():
            dominant_colors = [p.dominant_color for p in members]
            color_counts = {c: dominant_colors.count(c) for c in set(dominant_colors)}
            summary[cluster_id] = {
                "size": len(members),
                "members": [p.name for p in members],
                "color_distribution": color_counts,
                "centroid": self.centroids[cluster_id],
            }
        return summary

    def predict(self, profile: PersonProfile) -> int:
        """Predict cluster for a new profile."""
        if not self.centroids:
            raise RuntimeError("Must call fit() first")
        dists = [_distance(profile.vector, c) for c in self.centroids]
        return dists.index(min(dists))
