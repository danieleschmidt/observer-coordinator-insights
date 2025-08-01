# Autonomous SDLC Workflow Integration

This document describes the integration of Terragon's autonomous SDLC enhancement system with your existing CI/CD workflows.

## Overview

The autonomous system requires specific workflow integrations to enable continuous value discovery and execution. Since GitHub Actions workflows cannot be automatically created, this document provides templates and integration requirements.

## Required Workflow Files

### 1. Autonomous Value Discovery Workflow

**File**: `.github/workflows/autonomous-value-discovery.yml`

```yaml
name: Autonomous Value Discovery

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:
  push:
    branches: [main]

jobs:
  discover-value:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run Value Discovery
        run: |
          python value_discovery_engine.py
          
      - name: Update Backlog
        run: |
          git config --local user.email "autonomous@terragon-labs.com"
          git config --local user.name "Terragon Autonomous"
          git add BACKLOG.md .terragon/
          git diff --staged --quiet || git commit -m "Update autonomous backlog discovery"
          
      - name: Create PR for High-Value Items
        if: ${{ github.ref == 'refs/heads/main' }}
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: "Auto-execute high-value SDLC improvements"
          title: "[AUTONOMOUS] High-value SDLC enhancements"
          body: |
            ## Autonomous Value Execution
            
            This PR contains automatically discovered and executed high-value improvements:
            
            - 🔍 **Discovered via**: Continuous value discovery engine
            - 📊 **Scoring**: WSJF + ICE + Technical Debt analysis
            - ✅ **Quality Gates**: All tests passed, linting clean
            - 🤖 **Generated**: Terragon Autonomous SDLC System
            
            See BACKLOG.md for full value analysis and prioritization.
          branch: autonomous/value-execution
          labels: |
            autonomous
            enhancement
            terragon
```

### 2. Perpetual Execution Workflow

**File**: `.github/workflows/perpetual-execution.yml`

```yaml
name: Perpetual SDLC Execution

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:
    inputs:
      max_cycles:
        description: 'Maximum execution cycles'
        required: false
        default: '50'
      dry_run:
        description: 'Run in simulation mode'
        type: boolean
        default: false

jobs:
  perpetual-execution:
    runs-on: ubuntu-latest
    timeout-minutes: 240  # 4 hour maximum
    
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.TERRAGON_TOKEN || secrets.GITHUB_TOKEN }}
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          pip install pip-audit safety
      
      - name: Run Perpetual Executor
        run: |
          python perpetual_executor.py \
            --max-cycles ${{ github.event.inputs.max_cycles || '50' }} \
            ${{ github.event.inputs.dry_run == 'true' && '--dry-run' || '' }}
      
      - name: Generate Execution Report
        if: always()
        run: |
          echo "## Perpetual Execution Report" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          if [ -f .terragon/session_report_*.json ]; then
            latest_report=$(ls -t .terragon/session_report_*.json | head -1)
            echo "**Session Summary:**" >> $GITHUB_STEP_SUMMARY
            echo '```json' >> $GITHUB_STEP_SUMMARY
            cat "$latest_report" >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY
          fi
      
      - name: Commit Autonomous Changes
        run: |
          git config --local user.email "autonomous@terragon-labs.com"
          git config --local user.name "Terragon Autonomous"
          git add .
          git diff --staged --quiet || git commit -m "Autonomous SDLC enhancements
          
          🤖 Generated with Terragon Autonomous SDLC
          
          Co-Authored-By: Terragon <autonomous@terragon-labs.com>"
          git push origin main
```

### 3. Value Metrics Reporting Workflow

**File**: `.github/workflows/value-metrics.yml`

```yaml
name: Value Metrics Reporting

on:
  schedule:
    - cron: '0 1 * * 1'  # Weekly on Monday at 1 AM
  workflow_dispatch:

jobs:
  generate-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install matplotlib seaborn plotly
      
      - name: Generate Value Metrics Report
        run: |
          python -c "
          import json
          import matplotlib.pyplot as plt
          from datetime import datetime, timedelta
          
          # Load metrics
          with open('.terragon/value-metrics.json') as f:
              metrics = json.load(f)
          
          # Create visualization dashboard
          fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
          
          # Value delivery over time
          ax1.bar(['Security', 'Performance', 'Tech Debt', 'Quality'], 
                  [metrics['continuousDiscovery']['securityItems'],
                   metrics['continuousDiscovery']['performanceOpportunities'],
                   metrics['continuousDiscovery']['technicalDebtItems'],
                   metrics['continuousDiscovery']['itemsDiscovered'] - 
                   metrics['continuousDiscovery']['securityItems'] -
                   metrics['continuousDiscovery']['performanceOpportunities'] -
                   metrics['continuousDiscovery']['technicalDebtItems']])
          ax1.set_title('Value Items by Category')
          
          # Repository maturity
          maturity = metrics['repositoryMetrics']['maturityPercentage']
          ax2.pie([maturity, 100-maturity], labels=['Mature', 'Opportunity'], 
                  autopct='%1.1f%%', startangle=90)
          ax2.set_title(f'Repository Maturity: {maturity}%')
          
          # Execution success rate
          cycles = metrics['continuousDiscovery']['cyclesCompleted']
          executed = metrics['continuousDiscovery']['itemsExecuted']
          success_rate = (executed / cycles * 100) if cycles > 0 else 0
          ax3.bar(['Success Rate'], [success_rate], color='green')
          ax3.set_ylim(0, 100)
          ax3.set_ylabel('Percentage')
          ax3.set_title(f'Execution Success Rate: {success_rate:.1f}%')
          
          # Security posture
          security_score = metrics['repositoryMetrics']['securityScore']
          ax4.bar(['Security Score'], [security_score], color='red')
          ax4.set_ylim(0, 100)
          ax4.set_ylabel('Score')
          ax4.set_title(f'Security Posture: {security_score}/100')
          
          plt.tight_layout()
          plt.savefig('value-metrics-dashboard.png', dpi=300, bbox_inches='tight')
          print('Value metrics dashboard generated')
          "
      
      - name: Upload Metrics Dashboard
        uses: actions/upload-artifact@v3
        with:
          name: value-metrics-dashboard
          path: value-metrics-dashboard.png
          retention-days: 30
      
      - name: Create Metrics Issue
        uses: peter-evans/create-issue-from-file@v4
        with:
          title: "📊 Weekly Value Metrics Report"
          content-filepath: .terragon/value-metrics.json
          labels: |
            metrics
            autonomous
            weekly-report
```

## Integration Requirements

### Environment Variables

Add these secrets to your GitHub repository:

- `TERRAGON_TOKEN`: Personal access token with repo write access
- `SLACK_WEBHOOK` (optional): For notifications

### Repository Settings

1. **Branch Protection**: Configure main branch protection with:
   - Require status checks
   - Require branches to be up to date
   - Allow force pushes for autonomous commits

2. **Actions Permissions**: Enable:
   - Read and write permissions for GITHUB_TOKEN
   - Allow Actions to create pull requests

### File Structure

Ensure these directories exist:
```
.terragon/
├── config.yaml
├── value-metrics.json
├── execution.log
├── execution_history.jsonl
└── session_reports/
```

## Customization

### Scheduling

Adjust cron schedules based on your needs:
- **Aggressive**: Every hour (`0 * * * *`)
- **Moderate**: Every 6 hours (`0 */6 * * *`)
- **Conservative**: Daily (`0 2 * * *`)

### Risk Thresholds

Modify risk thresholds in `.terragon/config.yaml`:
```yaml
scoring:
  thresholds:
    minScore: 10      # Minimum score to execute
    maxRisk: 0.8      # Maximum risk tolerance
    securityBoost: 2.0 # Security priority multiplier
```

### Execution Limits

Control autonomous execution:
```yaml
execution:
  maxConcurrentTasks: 1
  testRequirements:
    minCoverage: 80
    performanceRegression: 5
```

## Monitoring and Observability

### Metrics Collection

The system automatically tracks:
- Value items discovered per category
- Execution success rates
- Cycle times and performance
- Repository maturity progression

### Alerting

Set up alerts for:
- Execution failures
- Security vulnerabilities discovered
- Performance regressions
- Coverage drops

### Dashboards

Create dashboards to monitor:
- Value delivery velocity
- Technical debt reduction
- Security posture improvement
- Code quality trends

## Rollback Procedures

### Automatic Rollback

The system automatically rolls back changes if:
- Tests fail after execution
- Linting fails
- Security scans fail
- Performance regresses beyond threshold

### Manual Rollback

To manually disable autonomous execution:
1. Set `enabled: false` in `.terragon/config.yaml`
2. Remove or disable the scheduled workflows
3. Revert any unwanted autonomous commits

## Best Practices

1. **Start Conservative**: Begin with `dry_run: true` and low cycle counts
2. **Monitor Closely**: Watch initial executions for unexpected behavior
3. **Gradual Scaling**: Increase automation gradually as confidence builds
4. **Regular Reviews**: Weekly review of autonomous decisions and outcomes
5. **Human Oversight**: Maintain human review for high-risk changes

## Troubleshooting

### Common Issues

1. **Workflow Permissions**: Ensure GITHUB_TOKEN has write permissions
2. **Dependency Installation**: Verify all required packages are in requirements.txt
3. **File Conflicts**: Check that .terragon/ directory is not in .gitignore
4. **Python Version**: Ensure Python 3.9+ is used consistently

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support

For issues with the autonomous system:
1. Check execution logs in `.terragon/execution.log`
2. Review session reports in `.terragon/session_reports/`
3. Examine value metrics for anomalies
4. Contact Terragon Labs support with log files