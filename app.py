"""
Resilio AI Platform — Streamlit Dashboard
=========================================
Module  : app.py
Run     : streamlit run aegis_rca/app.py
Purpose : Live telemetry visualisation + Incident Response Console.
          Integrates directly with diagnose.py for rule-based RCA.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config — must be the FIRST Streamlit call ──────────────────────────
st.set_page_config(
    page_title="Resilio AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import local RCA engine ──────────────────────────────────────────────────
from diagnose import diagnose_root_cause, DEFAULT_THRESHOLDS, RCAResult

# ════════════════════════════════════════════════════════════════════════════
# THEME INJECTION — dark cybersecurity palette via markdown + CSS
# ════════════════════════════════════════════════════════════════════════════

DARK_CSS = """
<style>
/* ── Root & background ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Segoe UI', 'JetBrains Mono', monospace;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Metric cards ──────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.78rem !important; letter-spacing: .05em; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
    background: #21262d;
    color: #f0f6fc;
    border: 1px solid #ff4b4b;
    border-radius: 6px;
    font-size: 0.88rem;
    letter-spacing: .04em;
    padding: 0.45rem 1.2rem;
    transition: background 0.2s, box-shadow 0.2s;
}
.stButton > button:hover {
    background: #3d0000;
    box-shadow: 0 0 12px #ff4b4b88;
    color: #ff4b4b;
}

/* ── Code blocks ───────────────────────────────────────────────── */
code, pre {
    background: #0d1117 !important;
    color: #39d353 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
    font-size: 0.82rem;
}

/* ── Section headers ───────────────────────────────────────────── */
h1 { color: #58a6ff !important; letter-spacing: .03em; }
h2 { color: #79c0ff !important; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
h3 { color: #a5d6ff !important; }

/* ── Divider ───────────────────────────────────────────────────── */
hr { border-color: #21262d !important; }

/* ── Alert overrides ───────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 6px; font-size: 0.9rem; }

/* ── Dataframe ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }

/* ── Sidebar label ─────────────────────────────────────────────── */
.sidebar-label {
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 2px;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE — persists across reruns within the same browser session
# ════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    """Initialise mutable session state on first load."""
    if "history" not in st.session_state:
        # Pre-populate 60 ticks of synthetic baseline telemetry
        now = datetime.utcnow()
        ts  = [now - timedelta(seconds=(60 - i) * 2) for i in range(60)]
        st.session_state.history = pd.DataFrame({
            "timestamp":       ts,
            "CPU_Usage":       np.clip(np.random.normal(42, 8, 60), 5, 99),
            "Memory_Usage":    np.clip(np.random.normal(55, 6, 60), 10, 99),
            "Network_Latency": np.clip(np.random.normal(80, 20, 60), 5, 499),
            "Error_Rate":      np.clip(np.random.normal(1.2, 0.6, 60), 0, 9),
        })
    if "outage_active"    not in st.session_state: st.session_state.outage_active    = False
    if "rca_result"       not in st.session_state: st.session_state.rca_result       = None
    if "incident_log"     not in st.session_state: st.session_state.incident_log     = []
    if "tick"             not in st.session_state: st.session_state.tick             = 0

_init_state()

# ════════════════════════════════════════════════════════════════════════════
# DATA GENERATION — one new telemetry tick per rerun
# ════════════════════════════════════════════════════════════════════════════

def _next_tick(outage: bool) -> dict:
    """
    Produce the next metric sample.
    Normal: small random walk from the previous values.
    Outage : inject a DDoS-style spike pattern.
    """
    prev = st.session_state.history.iloc[-1]

    if outage:
        # Simulate DDoS: latency spikes, error rate climbs, CPU surges
        cpu     = float(np.clip(prev["CPU_Usage"]       + random.uniform(8, 18),   0, 99))
        memory  = float(np.clip(prev["Memory_Usage"]    + random.uniform(1,  5),   0, 99))
        latency = float(np.clip(prev["Network_Latency"] + random.uniform(60, 150), 0, 999))
        err     = float(np.clip(prev["Error_Rate"]      + random.uniform(1,  4),   0, 30))
    else:
        # Normal ops: mean-reverting random walk
        cpu     = float(np.clip(prev["CPU_Usage"]       + random.gauss(0, 3),    5, 75))
        memory  = float(np.clip(prev["Memory_Usage"]    + random.gauss(0, 2),   10, 72))
        latency = float(np.clip(prev["Network_Latency"] + random.gauss(0, 12),   5, 180))
        err     = float(np.clip(prev["Error_Rate"]      + random.gauss(0, 0.3),  0, 4.5))

    return {
        "timestamp":       datetime.utcnow(),
        "CPU_Usage":       cpu,
        "Memory_Usage":    memory,
        "Network_Latency": latency,
        "Error_Rate":      err,
    }

# Append one tick and keep only the last 90 samples (3-minute window)
new_tick = _next_tick(st.session_state.outage_active)
st.session_state.history = pd.concat(
    [st.session_state.history, pd.DataFrame([new_tick])],
    ignore_index=True,
).tail(90)
st.session_state.tick += 1

latest = st.session_state.history.iloc[-1]

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — controls & system info
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🛡️ Resilio AI")
    st.caption("Predictive Infrastructure Intelligence")
    st.divider()

    st.markdown('<div class="sidebar-label">Auto-refresh interval</div>', unsafe_allow_html=True)
    refresh_sec = st.slider("", min_value=1, max_value=10, value=3, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sidebar-label">Threshold overrides</div>', unsafe_allow_html=True)

    cpu_crit  = st.number_input("CPU critical (%)",     value=DEFAULT_THRESHOLDS["CPU_Usage_critical"],      step=1.0)
    mem_crit  = st.number_input("Memory critical (%)",  value=DEFAULT_THRESHOLDS["Memory_Usage_critical"],   step=1.0)
    lat_crit  = st.number_input("Latency critical (ms)",value=DEFAULT_THRESHOLDS["Network_Latency_critical"],step=10.0)
    err_crit  = st.number_input("Error rate critical (%)",value=DEFAULT_THRESHOLDS["Error_Rate_critical"],   step=0.5)

    custom_thresholds = {
        **DEFAULT_THRESHOLDS,
        "CPU_Usage_critical":        cpu_crit,
        "Memory_Usage_critical":     mem_crit,
        "Network_Latency_critical":  lat_crit,
        "Error_Rate_critical":       err_crit,
    }

    st.divider()
    st.markdown('<div class="sidebar-label">System</div>', unsafe_allow_html=True)
    st.caption(f"Cluster : `prod-k8s-us-east-1`")
    st.caption(f"Node    : `worker-node-07`")
    st.caption(f"Ticks   : `{st.session_state.tick}`")
    st.caption(f"UTC     : `{datetime.utcnow().strftime('%H:%M:%S')}`")

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='text-align:center;'>🛡️ Resilio AI: Predictive Infrastructure Intelligence</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:0.85rem;margin-top:-10px;'>"
    "Real-time anomaly detection · Rule-based Root Cause Analysis · Automated Remediation"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Live KPI Metric Cards
# ════════════════════════════════════════════════════════════════════════════

st.markdown("## 📡 Live System Telemetry")

prev  = st.session_state.history.iloc[-2] if len(st.session_state.history) >= 2 else latest
col1, col2, col3, col4 = st.columns(4)

def _delta_color(val: float, warn: float, crit: float):
    
    if val >= crit: return "inverse"   # red arrow on metric delta
    if val >= warn: return "off"
    return "normal"

with col1:
    st.metric(
        "CPU Usage",
        f"{latest['CPU_Usage']:.1f} %",
        delta=f"{latest['CPU_Usage'] - prev['CPU_Usage']:+.1f}%",
        delta_color=_delta_color(latest["CPU_Usage"], DEFAULT_THRESHOLDS["CPU_Usage_high"], cpu_crit),
    )
with col2:
    st.metric(
        "Memory Usage",
        f"{latest['Memory_Usage']:.1f} %",
        delta=f"{latest['Memory_Usage'] - prev['Memory_Usage']:+.1f}%",
        delta_color=_delta_color(latest["Memory_Usage"], DEFAULT_THRESHOLDS["Memory_Usage_high"], mem_crit),
    )
with col3:
    st.metric(
        "Network Latency",
        f"{latest['Network_Latency']:.0f} ms",
        delta=f"{latest['Network_Latency'] - prev['Network_Latency']:+.0f} ms",
        delta_color=_delta_color(latest["Network_Latency"], DEFAULT_THRESHOLDS["Network_Latency_high"], lat_crit),
    )
with col4:
    st.metric(
        "Error Rate",
        f"{latest['Error_Rate']:.2f} %",
        delta=f"{latest['Error_Rate'] - prev['Error_Rate']:+.2f}%",
        delta_color=_delta_color(latest["Error_Rate"], DEFAULT_THRESHOLDS["Error_Rate_high"], err_crit),
    )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Time-series Charts
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("## 📈 Time-Series Telemetry  *(rolling 90-tick window)*")

chart_df = st.session_state.history.set_index("timestamp")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("#### CPU Usage (%)")
    st.line_chart(
        chart_df[["CPU_Usage"]],
        color=["#58a6ff"],
        use_container_width=True,
        height=200,
    )
    st.markdown("#### Memory Usage (%)")
    st.line_chart(
        chart_df[["Memory_Usage"]],
        color=["#bc8cff"],
        use_container_width=True,
        height=200,
    )

with chart_right:
    st.markdown("#### Network Latency (ms)")
    st.line_chart(
        chart_df[["Network_Latency"]],
        color=["#ff7b72"],
        use_container_width=True,
        height=200,
    )
    st.markdown("#### Error Rate (%)")
    st.line_chart(
        chart_df[["Error_Rate"]],
        color=["#ffa657"],
        use_container_width=True,
        height=200,
    )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Continuous Anomaly Check (runs every tick)
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("## 🔎 Isolation Forest — Live Anomaly Verdict")

# Lightweight inline score: flag when any metric crosses its HIGH threshold
def _is_anomalous(row: pd.Series, t: dict) -> bool:
    return (
        row["CPU_Usage"]       >= t["CPU_Usage_high"]        or
        row["Memory_Usage"]    >= t["Memory_Usage_high"]     or
        row["Network_Latency"] >= t["Network_Latency_high"]  or
        row["Error_Rate"]      >= t["Error_Rate_high"]
    )

is_anomaly = _is_anomalous(latest, custom_thresholds)

v_col1, v_col2 = st.columns([1, 3])
with v_col1:
    if is_anomaly:
        st.error("🚨  ANOMALY DETECTED")
    else:
        st.success("✅  NORMAL OPERATIONS")

with v_col2:
    if is_anomaly:
        live_result: RCAResult = diagnose_root_cause(pd.Series(latest), custom_thresholds)
        st.warning(
            f"**Diagnosis:** {live_result.diagnosis}  \n"
            f"**Severity:** `{live_result.severity}`  |  "
            f"**Confidence:** `{live_result.confidence}`  |  "
            f"**Trigger:** `{live_result.primary_metric}`"
        )
        # Log to incident list (deduplicate by 5-second buckets)
        bucket = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")[:-1]  # 10-sec bucket
        if not st.session_state.incident_log or st.session_state.incident_log[-1]["bucket"] != bucket:
            st.session_state.incident_log.append({
                "bucket":    bucket,
                "time":      datetime.utcnow().strftime("%H:%M:%S"),
                "diagnosis": live_result.diagnosis,
                "severity":  live_result.severity,
            })

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Incident Response Console
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("## 🚨 Incident Response Console")

console_left, console_right = st.columns([1, 2])

OUTAGE_REMEDIATION = OUTAGE_REMEDIATION = (
    "# 1. DEFENSE TIER 1: Network Layer Drop\n"
    "# Applying zero-trust policy to drop anomalous packets\n"
    "kubectl apply -f /k8s/security/ddos-packet-drop-policy.yaml && \\\n\n"
    
    "# 2. DEFENSE TIER 2: Load Absorption\n"
    "# Scaling ingress gateway to absorb volumetric impact\n"
    "kubectl scale deployment ingress-nginx --replicas=10 -n ingress-nginx && \\\n\n"
    
    "# 3. DEFENSE TIER 3: Incident Orchestration\n"
    "# Silencing redundant alerts for SRE teams\n"
    "curl -s -X POST http://alertmanager:9093/api/v2/silences \\\n"
    "  -H 'Content-Type: application/json' \\\n"
    "  -d '{\"matchers\":[{\"name\":\"alertname\",\"value\":\"HighLatency\"}],\n"
    "       \"startsAt\":\"now\",\"endsAt\":\"now+2h\",\"comment\":\"AEGIS auto-remediation active\"}'"
)

with console_left:
    st.markdown(
        "<p style='color:#8b949e;font-size:0.82rem;'>"
        "Inject a synthetic outage to observe how Resilio AI detects, diagnoses, "
        "and prescribes remediation in real time."
        "</p>",
        unsafe_allow_html=True,
    )

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("🔴  Simulate System Outage", use_container_width=True):
            st.session_state.outage_active = True
            st.session_state.rca_result    = None   # reset so fresh RCA runs below

    with btn_cols[1]:
        if st.button("🟢  Restore Normal Ops", use_container_width=True):
            st.session_state.outage_active = False
            st.session_state.rca_result    = None

    # Live indicator badge
    if st.session_state.outage_active:
        st.markdown(
            "<div style='margin-top:8px;padding:6px 12px;background:#3d0000;"
            "border:1px solid #ff4b4b;border-radius:6px;color:#ff4b4b;"
            "font-size:0.82rem;text-align:center;letter-spacing:.06em;'>"
            "⚡ OUTAGE SIMULATION ACTIVE</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='margin-top:8px;padding:6px 12px;background:#0d2818;"
            "border:1px solid #39d353;border-radius:6px;color:#39d353;"
            "font-size:0.82rem;text-align:center;letter-spacing:.06em;'>"
            "● SYSTEM NOMINAL</div>",
            unsafe_allow_html=True,
        )

with console_right:
    if st.session_state.outage_active:
        # ── Critical alert banner ─────────────────────────────────────────
        st.error(
            "🚨 **CRITICAL: Regime Shift Detected in Network Latency**  \n"
            "Isolation Forest confidence score: **−0.412** (threshold −0.15)  \n"
            "Triggered at: **" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC") + "**"
        )

        # ── Run RCA on latest live row ────────────────────────────────────
        outage_row = pd.Series({
            "CPU_Usage":       latest["CPU_Usage"],
            "Memory_Usage":    latest["Memory_Usage"],
            "Network_Latency": latest["Network_Latency"],
            "Error_Rate":      latest["Error_Rate"],
        })
        rca = diagnose_root_cause(outage_row, custom_thresholds)
        st.session_state.rca_result = rca

        # ── Diagnosis card ────────────────────────────────────────────────
        st.markdown(
            f"""
<div style='background:#161b22;border:1px solid #f0883e;border-radius:8px;padding:14px 18px;margin:8px 0;'>
  <div style='color:#f0883e;font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase;'>Root Cause Analysis</div>
  <div style='color:#ffa657;font-size:1.1rem;font-weight:600;margin:4px 0;'>{rca.diagnosis}</div>
  <div style='display:flex;gap:16px;margin-top:6px;'>
    <span style='background:#3d1a00;color:#f0883e;padding:2px 8px;border-radius:4px;font-size:0.75rem;'>
      SEVERITY: {rca.severity}
    </span>
    <span style='background:#1a1a3d;color:#79c0ff;padding:2px 8px;border-radius:4px;font-size:0.75rem;'>
      CONFIDENCE: {rca.confidence}
    </span>
    <span style='background:#1a3d1a;color:#56d364;padding:2px 8px;border-radius:4px;font-size:0.75rem;'>
      TRIGGER: {rca.primary_metric or "Multi-metric"}
    </span>
  </div>
  <div style='color:#8b949e;font-size:0.78rem;margin-top:8px;line-height:1.5;'>{rca.notes}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Remediation command block ─────────────────────────────────────
        st.markdown("**📋 Automated Remediation Command**")
        st.code(OUTAGE_REMEDIATION, language="bash")

        st.warning(
            "⚠️ Remediation command ready. Requires Kubernetes cluster-admin role. "
            "Review before executing in production.",
            icon="⚠️",
        )

    else:
        st.info(
            "Console is idle. Press **Simulate System Outage** to inject a "
            "synthetic DDoS-pattern incident and observe the full RCA pipeline.",
            icon="ℹ️",
        )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Incident Event Log
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.incident_log:
    st.divider()
    st.markdown("## 📋 Incident Event Log")
    log_df = pd.DataFrame(st.session_state.incident_log[-15:][::-1])  # newest first, max 15
    log_df.columns = ["Bucket", "Time (UTC)", "Diagnosis", "Severity"]
    st.dataframe(
        log_df[["Time (UTC)", "Diagnosis", "Severity"]],
        use_container_width=True,
        hide_index=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Raw telemetry table (collapsible)
# ════════════════════════════════════════════════════════════════════════════

with st.expander("🔬 Raw Telemetry Buffer  (last 20 ticks)", expanded=False):
    display_df = (
        st.session_state.history
        .tail(20)[::-1]
        .assign(timestamp=lambda d: d["timestamp"].dt.strftime("%H:%M:%S"))
        .rename(columns={
            "timestamp":       "Time (UTC)",
            "CPU_Usage":       "CPU (%)",
            "Memory_Usage":    "Memory (%)",
            "Network_Latency": "Latency (ms)",
            "Error_Rate":      "Error Rate (%)",
        })
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# AUTO-REFRESH  — rerun the whole app every N seconds
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    f"🔄 Auto-refreshing every **{refresh_sec}s** · "
    f"Tick `{st.session_state.tick}` · "
    f"Window: last `{len(st.session_state.history)}` samples · "
    "Resilio AI v1.0"
)
time.sleep(refresh_sec)
st.rerun()
