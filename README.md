# observer-coordinator-insights

Multi-agent orchestration for organizational analytics using the Insights Discovery wheel. Analyzes team composition across the four color quadrants (Red, Yellow, Green, Blue) and generates data-driven team recommendations.

## Features
- **PersonProfile**: Models individuals with Insights Discovery color scores
- **ClusterAnalyzer**: Groups profiles using k-means clustering (stdlib only)
- **TeamComposer**: Analyzes team balance, identifies gaps, recommends hires
- **Report generator**: JSON team analytics with insights and recommendations

## Usage

```python
from insights import PersonProfile, TeamComposer, generate_team_report

profiles = [
    PersonProfile("Alice", red=80, yellow=20, green=15, blue=30),
    PersonProfile("Bob", red=20, yellow=75, green=40, blue=25),
    PersonProfile("Carol", red=10, yellow=15, green=85, blue=50),
]

team = TeamComposer(profiles)
print(team.team_summary())
print(team.recommend_hire())

report = generate_team_report(profiles)
import json; print(json.dumps(report, indent=2))
```

## Color Quadrants

| Color | Energy | Traits |
|---|---|---|
| 🔴 Red | Fiery | Competitive, decisive, demanding |
| 🟡 Yellow | Sunshine | Enthusiastic, sociable, creative |
| 🟢 Green | Earth | Caring, patient, harmonious |
| 🔵 Blue | Cool | Analytical, precise, cautious |

## Testing

```bash
pytest tests/ -v  # 19 tests
```
