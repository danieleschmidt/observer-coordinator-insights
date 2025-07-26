# observer-coordinator-insights

[![Build Status](https://img.shields.io/github/actions/workflow/status/terragon-labs/observer-coordinator-insights/ci.yml?branch=main)](https://github.com/terragon-labs/observer-coordinator-insights/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/terragon-labs/observer-coordinator-insights)](https://coveralls.io/github/terragon-labs/observer-coordinator-insights)
[![License](https://img.shields.io/github/license/terragon-labs/observer-coordinator-insights)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

This project uses multi-agent orchestration to derive organizational analytics from Insights Discovery "wheel" data. It automatically clusters employees, simulates team compositions, and recommends cross-functional task forces.

## ✨ Key Features

*   **Automated Employee Clustering**: Uses K-means clustering on Insights Discovery data to group employees.
*   **Team Composition Simulation**: Simulates the potential dynamics and performance of different team compositions.
*   **Recommended Task Forces**: Suggests optimal cross-functional teams for specific projects.
*   **Embedded Visualization**: Integrates a Reveal-style cluster wheel to provide immediate visual value.

## 🛠️ Algorithm Transparency

We use **K-means clustering** for its computational efficiency and ease of interpretation, which is ideal for non-technical stakeholders. It creates distinct, non-overlapping clusters, providing clear groupings for initial analysis.

## 🔐 Privacy and Data Policy

*   **Security**: Insights Discovery data is sensitive. This tool ensures that all data is encrypted both at rest and in transit. No personally identifiable information (PII) is ever logged. Please refer to our organization's `SECURITY.md` for vulnerability reporting.
*   **Data Retention**: To comply with regulations like GDPR, all uploaded data is anonymized and purged after a default retention period of 180 days.

## ⚡ Quick Start

1.  Prepare your Insights Discovery data in CSV format.
2.  Configure the `observer-coordinator-insights.yml` file.
3.  Run the orchestrator to generate insights and view the embedded visualization.

## 📈 Roadmap

*   **v0.1.0**: Core functionality for employee clustering and team simulation.
*   **v0.2.0**: More advanced recommendation algorithms for task force composition.
*   **v0.3.0**: Integration with other HR and project management data sources.

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained.

## See Also

*   **[agentic-dev-orchestrator](../agentic-dev-orchestrator)**: Provides the core orchestration layer used by this tool.

## 📝 License

This project is licensed under the Apache-2.0 License.

## 📚 References

*   **Reveal Data**: [Product Page](https://www.revealdata.com/platform/processing-culling-filtering) and [Blog Post](https://www.revealdata.com/blog/adventure-ediscovery-with-cluster-wheel)
*   **Insights Discovery**: [Official Site](https://www.insights.com/products/insights-discovery/)
