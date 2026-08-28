"""
Resilio AI Platform — Root Cause Analysis Engine
================================================
Module  : diagnose.py
Purpose : Inspect an anomaly data row (pandas Series of system metrics),
          compare each metric against documented baseline thresholds, and
          return a structured diagnosis with a concrete Bash remediation
          command that an SRE can execute immediately.

Metric contract (features produced by the Isolation Forest pipeline)
--------------------------------------------------------------------
  CPU_Usage        float  — percentage utilisation (0–100)
  Memory_Usage     float  — percentage utilisation (0–100)
  Network_Latency  float  — round-trip latency in milliseconds
  Error_Rate       float  — application error rate as a percentage (0–100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline thresholds
# ---------------------------------------------------------------------------
# These values encode operational knowledge from post-mortem runbooks.
# Override them at runtime by passing a custom ``thresholds`` dict to
# ``diagnose_root_cause``.

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Compute
    "CPU_Usage_critical": 85.0,   # % — sustained above this → starvation risk
    "CPU_Usage_high": 70.0,        # % — elevated but recoverable

    # Memory
    "Memory_Usage_critical": 90.0, # % — OOM-kill territory
    "Memory_Usage_high": 75.0,     # % — possible leak / pressure

    # Network
    "Network_Latency_critical": 500.0,  # ms — severe degradation / DDoS
    "Network_Latency_high": 200.0,      # ms — SLA breach threshold

    # Error rate
    "Error_Rate_critical": 10.0,   # % — cascading failure / bad deploy
    "Error_Rate_high": 5.0,        # % — elevated; warrants investigation
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RCAResult:
    """
    Structured output returned by ``diagnose_root_cause``.

    Attributes
    ----------
    diagnosis       : Human-readable root-cause label.
    severity        : One of 'CRITICAL', 'HIGH', 'DEGRADED', or 'NORMAL'.
    confidence      : Qualitative confidence level ('HIGH', 'MEDIUM', 'LOW').
    primary_metric  : The metric that triggered the diagnosis (or None).
    remediation_cmd : Bash command string the on-call engineer should run.
    notes           : Optional free-text explanation for the runbook.
    """

    diagnosis: str
    severity: str
    confidence: str
    primary_metric: Optional[str]
    remediation_cmd: str
    notes: str = field(default="")

    def __str__(self) -> str:  # pretty-print for CLI / log lines
        lines = [
            "─" * 60,
            f"  Resilio AI Diagnosis   : {self.diagnosis}",
            f"  Severity              : {self.severity}",
            f"  Confidence            : {self.confidence}",
            f"  Primary Metric        : {self.primary_metric or 'N/A'}",
            f"  Remediation Command   : {self.remediation_cmd}",
        ]
        if self.notes:
            lines.append(f"  Notes                 : {self.notes}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: safe metric extraction
# ---------------------------------------------------------------------------

def _get_metric(row: pd.Series, key: str, default: float = 0.0) -> float:
    """
    Safely pull a float value from the anomaly row.

    Returns ``default`` (0.0) when the key is absent or the value is NaN,
    preventing the diagnosis from crashing on incomplete telemetry.
    """
    value = row.get(key, default)
    try:
        fval: float = float(value) if value is not None else default
        return default if pd.isna(fval) else fval
    except (TypeError, ValueError):
        logger.warning("Could not cast metric '%s' (value=%r) to float.", key, value)
        return default


# ---------------------------------------------------------------------------
# Core diagnosis function
# ---------------------------------------------------------------------------

def diagnose_root_cause(
    anomaly_data_row: pd.Series,
    thresholds: Optional[dict[str, float]] = None,
) -> RCAResult:
    """
    Perform rule-based root-cause analysis on a single anomaly observation.

    Parameters
    ----------
    anomaly_data_row : pd.Series
        One row from the anomaly dataset, indexed by metric name.
        Expected keys: 'CPU_Usage', 'Memory_Usage',
                        'Network_Latency', 'Error_Rate'.

    thresholds : dict[str, float], optional
        Override the default baseline thresholds (``DEFAULT_THRESHOLDS``).
        Useful for environment-specific tuning (e.g., GPU nodes, batch jobs).

    Returns
    -------
    RCAResult
        Structured diagnosis, severity rating, and ready-to-run
        Bash remediation command.

    Decision precedence (highest to lowest risk)
    --------------------------------------------
    1. Network critical  + high error rate  → DDoS attack
    2. Network critical  alone              → Network saturation / packet loss
    3. Memory critical                      → Memory leak in pod
    4. CPU critical      + high error rate  → Bad deploy / CPU starvation
    5. CPU critical      alone              → CPU starvation
    6. Error rate critical                  → Cascading failure / bad deploy
    7. Memory high       + CPU high         → Resource pressure (co-tenancy)
    8. Network high                         → Latency degradation
    9. Memory high                          → Memory pressure
    10. CPU high                            → CPU contention
    11. Error rate high                     → Elevated error rate
    12. (fallback)                          → Unclassified anomaly
    """

    # ── 0. Merge caller-supplied overrides with the defaults ─────────────────
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ── 1. Extract metric values from the row ────────────────────────────────
    cpu      = _get_metric(anomaly_data_row, "CPU_Usage")
    memory   = _get_metric(anomaly_data_row, "Memory_Usage")
    latency  = _get_metric(anomaly_data_row, "Network_Latency")
    err_rate = _get_metric(anomaly_data_row, "Error_Rate")

    logger.debug(
        "Diagnosing — CPU=%.1f%% MEM=%.1f%% LAT=%.1fms ERR=%.1f%%",
        cpu, memory, latency, err_rate,
    )

    # ── 2. Rule engine ────────────────────────────────────────────────────────
    #
    # Rules are ordered by operational impact; the *first* matching rule wins.
    # Each rule returns an RCAResult immediately, which keeps cyclomatic
    # complexity low and each branch independently testable.

    # --- Rule 1: Network critical + high error rate → DDoS -------------------
    if latency >= t["Network_Latency_critical"] and err_rate >= t["Error_Rate_high"]:
        return RCAResult(
            diagnosis       = "DDoS Attack suspected",
            severity        = "CRITICAL",
            confidence      = "HIGH",
            primary_metric  = "Network_Latency + Error_Rate",
            remediation_cmd = (
                "kubectl apply -f /k8s/network-policy-block-suspicious.yaml && "
                "kubectl scale deployment ingress-nginx --replicas=10 -n ingress-nginx"
            ),
            notes=(
                f"Network latency={latency:.1f}ms (≥{t['Network_Latency_critical']}ms), "
                f"Error rate={err_rate:.1f}% (≥{t['Error_Rate_high']}%). "
                "Activate DDoS runbook RB-NET-001. "
                "Consider enabling CDN-level rate limiting."
            ),
        )

    # --- Rule 2: Network critical alone → packet loss / saturation -----------
    if latency >= t["Network_Latency_critical"]:
        return RCAResult(
            diagnosis       = "Network Saturation / Packet Loss",
            severity        = "CRITICAL",
            confidence      = "HIGH",
            primary_metric  = "Network_Latency",
            remediation_cmd = (
                "systemctl restart networking && "
                "tc qdisc del dev eth0 root 2>/dev/null; "
                "kubectl rollout restart deployment/api-gateway"
            ),
            notes=(
                f"Network latency={latency:.1f}ms exceeds critical threshold "
                f"({t['Network_Latency_critical']}ms). "
                "Check NIC saturation, cable health, and upstream BGP routes."
            ),
        )

    # --- Rule 3: Memory critical → memory leak in pod ------------------------
    if memory >= t["Memory_Usage_critical"]:
        return RCAResult(
            diagnosis       = "Memory Leak in Pod",
            severity        = "CRITICAL",
            confidence      = "HIGH",
            primary_metric  = "Memory_Usage",
            remediation_cmd = (
                "kubectl rollout restart deployment/app-server && "
                "kubectl autoscale deployment app-server "
                "--cpu-percent=70 --min=2 --max=10"
            ),
            notes=(
                f"Memory usage={memory:.1f}% (≥{t['Memory_Usage_critical']}%). "
                "OOM-kill risk. Capture heap dump before restart: "
                "`kubectl exec <pod> -- jcmd 1 GC.heap_dump /tmp/heap.hprof`."
            ),
        )

    # --- Rule 4: CPU critical + high error rate → bad deploy -----------------
    if cpu >= t["CPU_Usage_critical"] and err_rate >= t["Error_Rate_high"]:
        return RCAResult(
            diagnosis       = "Bad Deploy / CPU Starvation under Error Cascade",
            severity        = "CRITICAL",
            confidence      = "HIGH",
            primary_metric  = "CPU_Usage + Error_Rate",
            remediation_cmd = (
                "kubectl rollout undo deployment/app-server && "
                "kubectl scale deployment app-server --replicas=8"
            ),
            notes=(
                f"CPU={cpu:.1f}% + Error rate={err_rate:.1f}%. "
                "Likely caused by a recent bad deploy triggering CPU-intensive "
                "error-retry loops. Roll back and re-validate canary."
            ),
        )

    # --- Rule 5: CPU critical alone → CPU starvation -------------------------
    if cpu >= t["CPU_Usage_critical"]:
        return RCAResult(
            diagnosis       = "CPU Starvation",
            severity        = "CRITICAL",
            confidence      = "HIGH",
            primary_metric  = "CPU_Usage",
            remediation_cmd = (
                "kubectl scale deployment app-server --replicas=5 && "
                "kubectl set resources deployment app-server "
                "--limits=cpu=2000m --requests=cpu=500m"
            ),
            notes=(
                f"CPU usage={cpu:.1f}% (≥{t['CPU_Usage_critical']}%). "
                "Scale horizontally and review CPU limits. "
                "Profile with: `kubectl exec <pod> -- py-spy top --pid 1`."
            ),
        )

    # --- Rule 6: Error rate critical → cascading failure / bad deploy --------
    if err_rate >= t["Error_Rate_critical"]:
        return RCAResult(
            diagnosis       = "Cascading Failure / Bad Deploy",
            severity        = "CRITICAL",
            confidence      = "MEDIUM",
            primary_metric  = "Error_Rate",
            remediation_cmd = (
                "kubectl rollout undo deployment/app-server && "
                "systemctl restart nginx && "
                "curl -s -X POST http://alertmanager:9093/api/v1/silences "
                "-d '{\"matchers\":[{\"name\":\"alertname\",\"value\":\"HighErrorRate\"}],"
                "\"startsAt\":\"now\",\"endsAt\":\"now+1h\",\"comment\":\"RCA in progress\"}'"
            ),
            notes=(
                f"Error rate={err_rate:.1f}% (≥{t['Error_Rate_critical']}%). "
                "Correlate with the most recent deployment in CI/CD pipeline. "
                "Check Sentry / ELK for dominant exception class."
            ),
        )

    # --- Rule 7: Memory high + CPU high → resource pressure ------------------
    if memory >= t["Memory_Usage_high"] and cpu >= t["CPU_Usage_high"]:
        return RCAResult(
            diagnosis       = "Resource Pressure — Co-tenancy / Noisy Neighbour",
            severity        = "HIGH",
            confidence      = "MEDIUM",
            primary_metric  = "Memory_Usage + CPU_Usage",
            remediation_cmd = (
                "kubectl scale deployment app-server --replicas=4 && "
                "kubectl label node $(kubectl get node -o name | head -1) "
                "dedicated=app-server --overwrite"
            ),
            notes=(
                f"Memory={memory:.1f}% + CPU={cpu:.1f}%. "
                "Possible noisy-neighbour effect. "
                "Consider node affinity / resource quotas at namespace level."
            ),
        )

    # --- Rule 8: Network high → latency degradation --------------------------
    if latency >= t["Network_Latency_high"]:
        return RCAResult(
            diagnosis       = "Network Latency Degradation",
            severity        = "HIGH",
            confidence      = "MEDIUM",
            primary_metric  = "Network_Latency",
            remediation_cmd = (
                "systemctl restart systemd-resolved && "
                "kubectl rollout restart deployment/api-gateway && "
                "ping -c 5 8.8.8.8 >> /var/log/aegis/network-diag.log"
            ),
            notes=(
                f"Network latency={latency:.1f}ms (≥{t['Network_Latency_high']}ms). "
                "Check DNS resolution times, service-mesh sidecar health, "
                "and inter-zone egress charges."
            ),
        )

    # --- Rule 9: Memory high → memory pressure -------------------------------
    if memory >= t["Memory_Usage_high"]:
        return RCAResult(
            diagnosis       = "Memory Pressure",
            severity        = "HIGH",
            confidence      = "MEDIUM",
            primary_metric  = "Memory_Usage",
            remediation_cmd = (
                "kubectl rollout restart deployment/app-server && "
                "echo 3 > /proc/sys/vm/drop_caches"
            ),
            notes=(
                f"Memory usage={memory:.1f}% (≥{t['Memory_Usage_high']}%). "
                "Monitor with: `kubectl top pod --sort-by=memory`. "
                "Investigate for gradual memory leak pattern over 24 h."
            ),
        )

    # --- Rule 10: CPU high → CPU contention ----------------------------------
    if cpu >= t["CPU_Usage_high"]:
        return RCAResult(
            diagnosis       = "CPU Contention",
            severity        = "HIGH",
            confidence      = "MEDIUM",
            primary_metric  = "CPU_Usage",
            remediation_cmd = (
                "kubectl scale deployment app-server --replicas=3 && "
                "renice -n -5 -p $(pgrep python)"
            ),
            notes=(
                f"CPU usage={cpu:.1f}% (≥{t['CPU_Usage_high']}%). "
                "Check for runaway cron jobs or background workers consuming cycles."
            ),
        )

    # --- Rule 11: Error rate high → elevated errors --------------------------
    if err_rate >= t["Error_Rate_high"]:
        return RCAResult(
            diagnosis       = "Elevated Error Rate",
            severity        = "DEGRADED",
            confidence      = "MEDIUM",
            primary_metric  = "Error_Rate",
            remediation_cmd = (
                "systemctl restart app-server && "
                "tail -n 500 /var/log/app/error.log | grep -E 'FATAL|ERROR' "
                ">> /var/log/aegis/error-digest.log"
            ),
            notes=(
                f"Error rate={err_rate:.1f}% (≥{t['Error_Rate_high']}%). "
                "Inspect application logs and recent config changes. "
                "Not yet at critical threshold—monitor trend closely."
            ),
        )

    # --- Rule 12: Fallback — anomaly detected but no clear single cause ------
    logger.warning(
        "No specific rule matched anomaly row. "
        "Metrics: CPU=%.1f, MEM=%.1f, LAT=%.1f, ERR=%.1f",
        cpu, memory, latency, err_rate,
    )
    return RCAResult(
        diagnosis       = "Unclassified Anomaly — Composite Metric Drift",
        severity        = "DEGRADED",
        confidence      = "LOW",
        primary_metric  = None,
        remediation_cmd = (
            "kubectl get events --sort-by=.lastTimestamp -A | tail -40 && "
            "journalctl -u app-server --since '10 minutes ago' -p err"
        ),
        notes=(
            f"Isolation Forest flagged this row as anomalous but all individual "
            f"metrics (CPU={cpu:.1f}%, MEM={memory:.1f}%, "
            f"LAT={latency:.1f}ms, ERR={err_rate:.1f}%) remain below their "
            "defined thresholds. Possible cause: an unusual *combination* of "
            "slightly-elevated metrics. Escalate to Tier-2 for manual review."
        ),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (python diagnose.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Ensure box-drawing characters render on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        pass
        #sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Synthetic anomaly scenarios to exercise all major rule branches
    test_cases: list[tuple[str, dict]] = [
        ("DDoS Attack",          {"CPU_Usage": 65.0, "Memory_Usage": 60.0, "Network_Latency": 620.0, "Error_Rate": 8.0}),
        ("Network Saturation",   {"CPU_Usage": 40.0, "Memory_Usage": 50.0, "Network_Latency": 550.0, "Error_Rate": 1.0}),
        ("Memory Leak",          {"CPU_Usage": 55.0, "Memory_Usage": 93.0, "Network_Latency": 80.0,  "Error_Rate": 2.0}),
        ("Bad Deploy",           {"CPU_Usage": 88.0, "Memory_Usage": 60.0, "Network_Latency": 90.0,  "Error_Rate": 7.0}),
        ("CPU Starvation",       {"CPU_Usage": 91.0, "Memory_Usage": 55.0, "Network_Latency": 100.0, "Error_Rate": 1.5}),
        ("Cascading Failure",    {"CPU_Usage": 60.0, "Memory_Usage": 65.0, "Network_Latency": 150.0, "Error_Rate": 12.0}),
        ("Resource Pressure",    {"CPU_Usage": 72.0, "Memory_Usage": 78.0, "Network_Latency": 120.0, "Error_Rate": 2.0}),
        ("Latency Degradation",  {"CPU_Usage": 45.0, "Memory_Usage": 50.0, "Network_Latency": 250.0, "Error_Rate": 1.0}),
        ("Memory Pressure",      {"CPU_Usage": 55.0, "Memory_Usage": 80.0, "Network_Latency": 90.0,  "Error_Rate": 1.0}),
        ("CPU Contention",       {"CPU_Usage": 74.0, "Memory_Usage": 55.0, "Network_Latency": 90.0,  "Error_Rate": 1.0}),
        ("Elevated Error Rate",  {"CPU_Usage": 50.0, "Memory_Usage": 55.0, "Network_Latency": 90.0,  "Error_Rate": 6.0}),
        ("Unclassified",         {"CPU_Usage": 40.0, "Memory_Usage": 50.0, "Network_Latency": 90.0,  "Error_Rate": 1.0}),
    ]

    print("\n" + "═" * 60)
    print("  Resilio AI  —  Smoke Test Suite")
    print("═" * 60)

    for label, metrics in test_cases:
        row = pd.Series(metrics)
        result = diagnose_root_cause(row)
        print(f"\n[TEST] {label}")
        print(result)
