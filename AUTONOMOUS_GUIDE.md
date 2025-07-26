# Autonomous Backlog Management System

## Overview

This system implements an autonomous senior coding assistant that discovers, prioritizes, and executes backlog items using WSJF (Weighted Shortest Job First) scoring and TDD micro-cycles.

## Quick Start

1. **View current backlog status:**
   ```bash
   python3 run_autonomous.py --status
   ```

2. **Run discovery to find TODOs/FIXMEs:**
   ```bash
   python3 run_autonomous.py --discovery
   ```

3. **Generate metrics report:**
   ```bash
   python3 run_autonomous.py --metrics
   ```

4. **Execute a single item:**
   ```bash
   python3 run_autonomous.py --item BL-001
   ```

5. **Run full autonomous mode:**
   ```bash
   python3 run_autonomous.py
   ```

## System Architecture

### Core Components

1. **BacklogManager** (`backlog_manager.py`)
   - Loads and manages `backlog.yml`
   - WSJF scoring with aging multipliers
   - Continuous discovery of TODOs/FIXMEs/failing tests

2. **ExecutionEngine** (`execution_engine.py`) 
   - TDD micro-cycles (RED → GREEN → REFACTOR)
   - Security checklist verification
   - Documentation updates
   - CI gate checks

3. **MetricsReporter** (`metrics_reporter.py`)
   - Comprehensive status reports
   - JSON and Markdown output
   - Backlog health analysis
   - Recommendations engine

4. **AutonomousOrchestrator** (`autonomous_orchestrator.py`)
   - Main "DO UNTIL DONE" execution loop
   - Scope and permission controls
   - Risk escalation for high-risk items

## WSJF Scoring

Items are prioritized using WSJF:
```
WSJF = (Value + Time Criticality + Risk Reduction) / Effort × Aging Multiplier
```

- **Value:** Business/user value (1-2-3-5-8-13)
- **Time Criticality:** Urgency of delivery (1-2-3-5-8-13)  
- **Risk Reduction:** Risk mitigation value (1-2-3-5-8-13)
- **Effort:** Implementation effort (1-2-3-5-8-13)
- **Aging Multiplier:** Boosts stale items (1.0-2.0)

## Backlog States

Items flow through these states:
```
NEW → REFINED → READY → DOING → PR → DONE/BLOCKED
```

- **NEW:** Just discovered
- **REFINED:** Requirements clarified
- **READY:** Ready for execution
- **DOING:** Currently being worked
- **PR:** Ready for pull request
- **DONE:** Completed
- **BLOCKED:** Requires intervention

## TDD Micro-Cycle

Each ready item goes through:

1. **RED:** Write failing test
2. **GREEN:** Make test pass (manual coding)
3. **REFACTOR:** Clean up code
4. **SECURITY:** Security checklist
5. **DOCS:** Update documentation  
6. **CI:** Lint, test, type checks
7. **PR:** Prepare pull request

## Scope Controls

The system respects scope defined in `.automation-scope.yaml`:

- **Default scope:** Current repository only
- **Extended scope:** Requires approval or manifest
- **High-risk items:** Always require human approval
- **Operation limits:** Configurable safety limits

## Security Features

- Automatic security checklist for relevant items
- No PII in logs or commits
- Secrets via environment variables only
- Audit trail for all operations
- Risk-tier classification

## Reports and Metrics

Status reports include:
- Backlog health metrics
- WSJF score distribution
- Cycle time analysis  
- Risk and blocker identification
- Test coverage tracking
- Actionable recommendations

Reports are saved to `docs/status/` in both JSON and Markdown formats.

## Configuration

### Backlog Configuration (`backlog.yml`)
```yaml
items:
  - id: "BL-001"
    title: "Feature title"
    type: "feature|bug|security|infrastructure"
    description: "Detailed description"
    acceptance_criteria:
      - "Criterion 1"
      - "Criterion 2"
    effort: 5           # 1-2-3-5-8-13
    value: 8            # 1-2-3-5-8-13
    time_criticality: 3 # 1-2-3-5-8-13
    risk_reduction: 5   # 1-2-3-5-8-13
    status: "READY"
    risk_tier: "low|medium|high"
    created_at: "2025-07-26T00:00:00Z"
```

### Scope Configuration (`.automation-scope.yaml`)
```yaml
allowed_paths:
  - "../sibling-repo"
  
require_approval:
  - "cross_repo_changes"
  - "ci_config_changes"
  
limits:
  max_operations_per_session: 100
```

## Best Practices

1. **Start small:** Begin with low-risk, well-defined items
2. **Review regularly:** Check reports and adjust priorities
3. **Maintain quality:** Ensure acceptance criteria are clear
4. **Monitor scope:** Review automation scope periodically  
5. **Human oversight:** Always review high-risk changes

## Safety Features

- **Operation limits:** Prevents runaway execution
- **Graceful shutdown:** Handles interrupts cleanly
- **Risk escalation:** High-risk items require approval
- **Audit trail:** All operations are logged
- **Scope boundaries:** Respects configured limits

## Troubleshooting

### Common Issues

1. **No ready items:** Items may need refinement or risk assessment
2. **High failure rate:** Check CI configuration and test setup
3. **Scope violations:** Review `.automation-scope.yaml`
4. **High risk escalations:** Ensure proper human review process

### Debug Commands

```bash
# Check current status
python3 run_autonomous.py --status

# Run discovery only  
python3 run_autonomous.py --discovery

# Generate detailed metrics
python3 run_autonomous.py --metrics

# Test single item execution
python3 run_autonomous.py --item <ITEM_ID>
```

## Integration

The system integrates with:
- **Git:** Repository state and change tracking
- **CI/CD:** Test execution and quality gates
- **Testing frameworks:** pytest, npm test, cargo test, go test
- **Linting tools:** eslint, ruff, pylint, clippy
- **Type checkers:** mypy, TypeScript, cargo check

## Continuous Improvement

The system includes meta-tasks for:
- Process optimization
- WSJF weight tuning  
- Workflow improvements
- Quality metric enhancement
- Performance optimization

When the backlog is empty, it evaluates system health and proposes improvements.

## License

Apache-2.0 License