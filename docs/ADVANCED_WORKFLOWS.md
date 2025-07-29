# Advanced GitHub Workflows

This document contains additional GitHub Actions workflows that can enhance the repository's SDLC maturity. These workflows require repository admin permissions to add.

## Security and Compliance Workflows

### 1. CodeQL Advanced Security (`codeql.yml`)

Advanced security scanning with extended query sets:

```yaml
name: "CodeQL Advanced Security"

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '17 4 * * 2'

jobs:
  analyze:
    name: Analyze (${{ matrix.language }})
    runs-on: ubuntu-latest
    timeout-minutes: 360
    permissions:
      security-events: write
      packages: read
      actions: read
      contents: read

    strategy:
      fail-fast: false
      matrix:
        include:
        - language: python
          build-mode: none

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        build-mode: ${{ matrix.build-mode }}
        queries: +security-extended,security-and-quality

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
      with:
        category: "/language:${{matrix.language}}"
```

### 2. Dependency Review (`dependency-review.yml`)

Scans dependencies for vulnerabilities in PRs:

```yaml
name: 'Dependency Review'
on: [pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: 'Checkout Repository'
        uses: actions/checkout@v4
        
      - name: 'Dependency Review'
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          allow-licenses: GPL-3.0-only, GPL-3.0-or-later, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Apache-2.0
          fail-on-scopes: runtime, development
```

### 3. OSSF Scorecard (`scorecard.yml`)

Supply chain security assessment:

```yaml
name: Scorecard supply-chain security
on:
  branch_protection_rule:
  schedule:
    - cron: '32 2 * * 6'
  push:
    branches: [ "main" ]

permissions: read-all

jobs:
  analysis:
    name: Scorecard analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      id-token: write
      contents: read

    steps:
      - name: "Checkout code"
        uses: actions/checkout@v4
        with:
          persist-credentials: false

      - name: "Run analysis"
        uses: ossf/scorecard-action@v2.3.1
        with:
          results_file: results.sarif
          results_format: sarif
          publish_results: true

      - name: "Upload artifact"
        uses: actions/upload-artifact@v4
        with:
          name: SARIF file
          path: results.sarif
          retention-days: 5

      - name: "Upload to code-scanning"
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### 4. SBOM Generation (`sbom.yml`)

Software Bill of Materials generation:

```yaml
name: Generate SBOM

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  release:
    types: [published]
  schedule:
    - cron: '0 6 * * 1'

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install cyclonedx-bom syft

      - name: Generate Python SBOM with CycloneDX
        run: |
          cyclonedx-py -o sbom-cyclonedx.json --format json
          cyclonedx-py -o sbom-cyclonedx.xml --format xml

      - name: Generate SPDX SBOM with Syft
        run: |
          curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
          syft packages dir:. -o spdx-json=sbom-spdx.json
          syft packages dir:. -o spdx-tag-value=sbom-spdx.spdx

      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sbom-files
          path: |
            sbom-*.json
            sbom-*.xml
            sbom-*.spdx
          retention-days: 90
```

### 5. SLSA Provenance (`slsa-generic.yml`)

Supply chain provenance generation:

```yaml
name: SLSA Generic Generator
on:
  workflow_dispatch:
  release:
    types: [created]

permissions:
  actions: read
  id-token: write
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digests: ${{ steps.hash.outputs.digests }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install build wheel

      - name: Build distribution
        run: |
          python -m build
          
      - name: Generate subject for provenance
        id: hash
        run: |
          set -euo pipefail
          cd dist/
          echo "digests=$(sha256sum * | base64 -w0)" >> "$GITHUB_OUTPUT"

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-artifacts
          path: dist/
          if-no-files-found: error
          retention-days: 90

  provenance:
    if: startsWith(github.ref, 'refs/tags/')
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.digests }}"
      upload-assets: true
```

## Automation Workflows

### 6. Auto-merge Dependabot (`auto-merge-dependabot.yml`)

Safely auto-merge dependency updates:

```yaml
name: Auto-merge Dependabot PRs

on:
  pull_request_target:
    types: [opened, reopened, synchronize]

permissions:
  contents: write
  pull-requests: write
  checks: read

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    steps:
      - name: Get Dependabot PR metadata
        id: dependabot-metadata
        uses: dependabot/fetch-metadata@v1
        with:
          github-token: "${{ secrets.GITHUB_TOKEN }}"

      - name: Auto-approve safe updates
        if: steps.dependabot-metadata.outputs.update-type == 'version-update:semver-patch'
        run: gh pr review --approve "$PR_URL"
        env:
          PR_URL: ${{github.event.pull_request.html_url}}
          GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}

      - name: Enable auto-merge for patches
        if: steps.dependabot-metadata.outputs.update-type == 'version-update:semver-patch'
        run: gh pr merge --auto --rebase "$PR_URL"
        env:
          PR_URL: ${{github.event.pull_request.html_url}}
          GITHUB_TOKEN: ${{secrets.GITHUB_TOKEN}}
```

### 7. Stale Issue Management (`stale.yml`)

Manage inactive issues and PRs:

```yaml
name: Mark stale issues and pull requests

on:
  schedule:
  - cron: '30 1 * * *'
  workflow_dispatch:

permissions:
  issues: write
  pull-requests: write

jobs:
  stale:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/stale@v9
      with:
        repo-token: ${{ secrets.GITHUB_TOKEN }}
        days-before-stale: 60
        days-before-close: 7
        days-before-pr-close: 14
        stale-issue-label: 'stale'
        stale-pr-label: 'stale'
        exempt-issue-labels: 'enhancement,bug,good first issue,help wanted,blocked,security'
        exempt-pr-labels: 'enhancement,bug,security,ready-to-merge,work-in-progress'
        stale-issue-message: |
          This issue has been automatically marked as stale because it has not had
          recent activity. It will be closed if no further activity occurs.
        stale-pr-message: |
          This pull request has been automatically marked as stale because it has not had
          recent activity. It will be closed if no further activity occurs.
```

### 8. Auto-labeler (`labeler.yml`)

The workflow to use the `.github/labeler.yml` configuration:

```yaml
name: Labeler
on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/labeler@v5
      with:
        repo-token: "${{ secrets.GITHUB_TOKEN }}"
        configuration-path: .github/labeler.yml
        sync-labels: true
```

## Installation Instructions

To add these workflows to your repository:

1. **Create the `.github/workflows/` directory** if it doesn't exist
2. **Copy the desired workflow files** from above into separate `.yml` files
3. **Customize the configurations** as needed for your repository
4. **Ensure you have the necessary permissions** or ask a repository admin to add them
5. **Test the workflows** by triggering their conditions (PRs, pushes, etc.)

## Benefits

Adding these workflows will provide:

- **🔒 Enhanced Security**: Automated vulnerability scanning and supply chain security
- **🤖 Intelligent Automation**: Smart dependency management and issue handling  
- **📊 Compliance Monitoring**: OSSF Scorecard and security posture tracking
- **🔍 Quality Assurance**: Comprehensive code analysis and dependency review
- **⚡ Developer Productivity**: Automated labeling and stale issue management

## Notes

- Some workflows require **GitHub Advanced Security** features
- **Repository admin permissions** are needed to add workflow files
- **Customize the configurations** to match your project's specific needs
- **Test thoroughly** before enabling auto-merge features