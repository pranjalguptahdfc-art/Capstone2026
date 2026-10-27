"""
analytics_view.py
-----------------
Streamlit Analytics Dashboard page for Banana Bunch Ripeness & Quality Control.
Displays:
  - KPI summary metrics (Total Bunches, Approved, Denied, Avg Ripeness, Total Bananas)
  - Past 30 Days trend charts (Approval vs Denial, Daily Ripeness %, Color Composition)
  - Ripeness Stage & Decision Distribution charts
  - Rejection / Denial root cause diagnostics
  - Comprehensive timestamped inspection log tables with search and filters
  - CSV export and testing utilities
"""

import io
from datetime import datetime, timedelta
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.analytics_store import (
    load_inspections,
    get_summary_metrics,
    log_inspection,
    seed_demo_data,
    LOG_FILE,
)

# Color constants
COLOR_APPROVED = "#27ae60"  # Green
COLOR_DENIED = "#e74c3c"    # Red
COLOR_SEMI = "#2980b9"      # Blue
COLOR_RIPE = "#f39c12"      # Gold / Amber
COLOR_UNRIPE = "#2ecc71"    # Light Green
COLOR_OVERRIPE = "#c0392b"  # Dark Red


def render_analytics_page():
    """Main entry point for rendering the Analytics page in Streamlit."""
    st.title("📊 Banana Ripeness & Quality Analytics")
    st.caption(
        "Automated telemetry and quality control logs recorded from the detection pipeline. "
        "Tracking bunch approvals, rejections, average ripeness, and color trends over the past 30 days."
    )

    # ── Sidebar / Filter Controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Analytics Filters")
        time_filter = st.selectbox(
            "Time Range",
            ["Past 30 Days", "Past 14 Days", "Past 7 Days", "Today", "All Time", "Custom Range"],
            index=0,
        )

        days_val = 30
        custom_start, custom_end = None, None

        if time_filter == "Past 30 Days":
            days_val = 30
        elif time_filter == "Past 14 Days":
            days_val = 14
        elif time_filter == "Past 7 Days":
            days_val = 7
        elif time_filter == "Today":
            days_val = 1
        elif time_filter == "All Time":
            days_val = None
        elif time_filter == "Custom Range":
            days_val = None
            today = datetime.now().date()
            custom_range = st.date_input(
                "Select Date Range",
                value=(today - timedelta(days=30), today),
                max_value=today,
            )
            if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
                custom_start, custom_end = custom_range

        decision_filter = st.radio(
            "Decision Filter",
            ["All", "Approved Only", "Denied Only"],
            index=0,
        )

        stage_filter = st.multiselect(
            "Ripeness Stage",
            ["Ripe", "Semi-Ripe", "Unripe", "Overripe"],
            default=["Ripe", "Semi-Ripe", "Unripe", "Overripe"],
        )

        st.divider()
        st.subheader("Data Management")
        if st.button("🧪 Simulate Test Inspection", help="Simulate logging an inspection right now"):
            _simulate_live_inspection()
            st.success("Test inspection logged successfully!")
            st.rerun()

        if st.button("🔄 Reset 30-Day Demo Data", help="Restore fresh 30-day simulated history"):
            seed_demo_data(force=True)
            st.success("Data reset to 30-day baseline!")
            st.rerun()

    # ── Load Data ──────────────────────────────────────────────────────────────
    df_raw = load_inspections(days=None)  # load all to apply filters accurately

    if df_raw.empty:
        st.warning("No inspection logs found. Seeding 30-day demo data...")
        seed_demo_data(force=True)
        df_raw = load_inspections(days=None)

    # Apply date filter
    now = datetime.now()
    if days_val is not None:
        cutoff = now - timedelta(days=days_val)
        df = df_raw[df_raw["timestamp"] >= cutoff].copy()
    elif custom_start and custom_end:
        df = df_raw[
            (df_raw["timestamp"].dt.date >= custom_start)
            & (df_raw["timestamp"].dt.date <= custom_end)
        ].copy()
    else:
        df = df_raw.copy()

    # Apply category filters for dashboard views
    df_filtered = df.copy()
    if decision_filter == "Approved Only":
        df_filtered = df_filtered[df_filtered["decision"] == "Approved"]
    elif decision_filter == "Denied Only":
        df_filtered = df_filtered[df_filtered["decision"] == "Denied"]

    if stage_filter:
        df_filtered = df_filtered[df_filtered["ripeness_stage"].isin(stage_filter)]

    # Compute KPIs from current active slice
    metrics = get_summary_metrics(df)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    _render_kpi_cards(metrics, days_val or 30)

    st.markdown("---")

    # ── Tabs for Analytics Views ───────────────────────────────────────────────
    tab_overview, tab_ripeness, tab_denials, tab_tables = st.tabs([
        "📈 Approval & Quality Trends",
        "🍌 Ripeness & Color Progression",
        "⚠️ Quality Gates & Rejection Diagnostics",
        "📋 Timestamped Inspection Logs",
    ])

    with tab_overview:
        _render_approval_trends_tab(df)

    with tab_ripeness:
        _render_ripeness_progression_tab(df)

    with tab_denials:
        _render_denials_tab(df)

    with tab_tables:
        _render_data_tables_tab(df, df_filtered)


# ──────────────────────────────────────────────────────────────────────────────
# KPI Metrics Component
# ──────────────────────────────────────────────────────────────────────────────

def _render_kpi_cards(metrics: dict, days: int):
    """Render top-level executive metrics."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"<div style='background:#f8f9fa; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='font-size:12px; font-weight:600; color:#64748b; text-transform:uppercase;'>Total Bunches</div>"
            f"<div style='font-size:28px; font-weight:bold; color:#0f172a; margin:4px 0;'>{metrics['total_bunches']:,}</div>"
            f"<div style='font-size:12px; color:#64748b;'>{metrics['total_bananas']:,} bananas scanned</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='font-size:12px; font-weight:600; color:#166534; text-transform:uppercase;'>Approved Bunches</div>"
            f"<div style='font-size:28px; font-weight:bold; color:#15803d; margin:4px 0;'>{metrics['approved_count']:,}</div>"
            f"<div style='font-size:12px; font-weight:600; color:#16a34a;'>{metrics['approval_rate']}% Approval Rate</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"<div style='background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='font-size:12px; font-weight:600; color:#991b1b; text-transform:uppercase;'>Denied Bunches</div>"
            f"<div style='font-size:28px; font-weight:bold; color:#b91c1c; margin:4px 0;'>{metrics['denied_count']:,}</div>"
            f"<div style='font-size:12px; font-weight:600; color:#dc2626;'>{metrics['denial_rate']}% Rejection Rate</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"<div style='background:#fffbeb; border:1px solid #fef3c7; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='font-size:12px; font-weight:600; color:#92400e; text-transform:uppercase;'>Avg Ripeness</div>"
            f"<div style='font-size:28px; font-weight:bold; color:#b45309; margin:4px 0;'>{metrics['avg_ripeness']}%</div>"
            f"<div style='font-size:12px; color:#78350f;'>Dominant: <b>{metrics['dominant_stage']}</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"<div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; padding:14px; text-align:center;'>"
            f"<div style='font-size:12px; font-weight:600; color:#0369a1; text-transform:uppercase;'>Bunch Density</div>"
            f"<div style='font-size:28px; font-weight:bold; color:#0284c7; margin:4px 0;'>{metrics['avg_bananas_per_bunch']:.1f}</div>"
            f"<div style='font-size:12px; color:#0369a1;'>Avg bananas per bunch</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Approval Trends Component
# ──────────────────────────────────────────────────────────────────────────────

def _render_approval_trends_tab(df: pd.DataFrame):
    """Render approval and denial volume and rates over time."""
    st.subheader("Inspection Decisions Over Time (Approved vs Denied)")

    if df.empty:
        st.info("No data in selected date range.")
        return

    # Aggregate by date
    df_daily = df.copy()
    df_daily["date"] = df_daily["timestamp"].dt.date

    # Pivot count by decision
    daily_counts = (
        df_daily.groupby(["date", "decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Approved" not in daily_counts.columns:
        daily_counts["Approved"] = 0
    if "Denied" not in daily_counts.columns:
        daily_counts["Denied"] = 0

    daily_counts["Total"] = daily_counts["Approved"] + daily_counts["Denied"]
    daily_counts["Approval_Rate"] = (
        (daily_counts["Approved"] / daily_counts["Total"] * 100).round(1)
    )

    # Melt for Altair stacked bar chart
    melted = daily_counts.melt(
        id_vars=["date", "Total", "Approval_Rate"],
        value_vars=["Approved", "Denied"],
        var_name="Decision",
        value_name="Count",
    )
    melted["date_str"] = melted["date"].astype(str)

    # Altair Stacked Bar Chart
    bar_chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("date_str:N", title="Inspection Date", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Count:Q", title="Bunches Inspected"),
            color=alt.Color(
                "Decision:N",
                scale=alt.Scale(
                    domain=["Approved", "Denied"],
                    range=[COLOR_APPROVED, COLOR_DENIED],
                ),
                legend=alt.Legend(title="Decision", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("date_str:N", title="Date"),
                alt.Tooltip("Decision:N", title="Decision"),
                alt.Tooltip("Count:Q", title="Bunches"),
                alt.Tooltip("Total:Q", title="Daily Total"),
                alt.Tooltip("Approval_Rate:Q", title="Approval Rate %"),
            ],
        )
        .properties(height=340)
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # Trend line for Daily Approval Rate %
    st.subheader("Daily Quality Approval Rate %")
    rate_chart = (
        alt.Chart(daily_counts)
        .mark_line(point=True, color="#10b981", strokeWidth=3)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "Approval_Rate:Q",
                title="Approval Rate (%)",
                scale=alt.Scale(domain=[50, 100]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Approved:Q", title="Approved"),
                alt.Tooltip("Denied:Q", title="Denied"),
                alt.Tooltip("Approval_Rate:Q", title="Approval Rate (%)"),
            ],
        )
        .properties(height=220)
    )

    # Target threshold line at 85%
    target_rule = (
        alt.Chart(pd.DataFrame({"y": [85]}))
        .mark_rule(color="#94a3b8", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    st.altair_chart(rate_chart + target_rule, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Ripeness & Color Progression Component
# ──────────────────────────────────────────────────────────────────────────────

def _render_ripeness_progression_tab(df: pd.DataFrame):
    """Render Ripeness Score trends and Color composition."""
    st.subheader("Average Ripeness Score (%) Over Time")

    if df.empty:
        st.info("No data in selected date range.")
        return

    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.date

    # Daily averages
    daily_stats = (
        df_copy.groupby("date")
        .agg(
            avg_ripeness=("avg_ripeness_pct", "mean"),
            avg_green=("green_pct", "mean"),
            avg_yellow=("yellow_pct", "mean"),
            avg_brown=("brown_pct", "mean"),
            bunch_count=("bunch_id", "count"),
        )
        .reset_index()
    )
    daily_stats["avg_ripeness"] = daily_stats["avg_ripeness"].round(1)

    # Daily ripeness line chart
    ripeness_chart = (
        alt.Chart(daily_stats)
        .mark_area(
            line={"color": "#d97706", "width": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#fbbf24", offset=0),
                    alt.GradientStop(color="#fef3c7", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "avg_ripeness:Q",
                title="Average Ripeness Score (%)",
                scale=alt.Scale(domain=[20, 100]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("avg_ripeness:Q", title="Avg Ripeness (%)"),
                alt.Tooltip("bunch_count:Q", title="Bunches Tested"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(ripeness_chart, use_container_width=True)

    # Color breakdown over time (Green %, Yellow %, Brown %)
    st.subheader("Daily Color Spectrum Composition")
    st.caption("Tracking chlorophyll breakdown (Green), carotenoid emergence (Yellow), and senescence spotting (Brown).")

    color_melted = daily_stats.melt(
        id_vars=["date"],
        value_vars=["avg_green", "avg_yellow", "avg_brown"],
        var_name="Color_Metric",
        value_name="Percentage",
    )
    color_melted["Percentage"] = color_melted["Percentage"].round(1)

    color_label_map = {
        "avg_green": "Green (Unripe)",
        "avg_yellow": "Yellow (Ripe)",
        "avg_brown": "Brown (Overripe)",
    }
    color_melted["Color_Label"] = color_melted["Color_Metric"].map(color_label_map)

    color_chart = (
        alt.Chart(color_melted)
        .mark_line(strokeWidth=2.5, point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Percentage:Q", title="Average Surface Area (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Color_Label:N",
                scale=alt.Scale(
                    domain=["Green (Unripe)", "Yellow (Ripe)", "Brown (Overripe)"],
                    range=["#27ae60", "#f1c40f", "#8b4513"],
                ),
                legend=alt.Legend(title="Pigment", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Color_Label:N", title="Pigment"),
                alt.Tooltip("Percentage:Q", title="Avg %"),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(color_chart, use_container_width=True)

    # Ripeness Stage Breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ripeness Stage Share")
        stage_counts = df["ripeness_stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]

        stage_pie = (
            alt.Chart(stage_counts)
            .mark_arc(innerRadius=45)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "Stage:N",
                    scale=alt.Scale(
                        domain=["Ripe", "Semi-Ripe", "Unripe", "Overripe"],
                        range=[COLOR_RIPE, COLOR_SEMI, COLOR_UNRIPE, COLOR_OVERRIPE],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["Stage:N", "Count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(stage_pie, use_container_width=True)

    with col2:
        st.subheader("Quality Status Share")
        status_counts = df["bunch_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        status_bar = (
            alt.Chart(status_counts)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Count:Q", title="Bunches"),
                y=alt.Y("Status:N", title="Status", sort="-x"),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["ACCEPTABLE", "SEMI-RIPE", "HIGH RISK"],
                        range=[COLOR_APPROVED, COLOR_SEMI, COLOR_DENIED],
                    ),
                ),
                tooltip=["Status:N", "Count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(status_bar, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Denials Diagnostics Component
# ──────────────────────────────────────────────────────────────────────────────

def _render_denials_tab(df: pd.DataFrame):
    """Render rejection diagnostics and failure reasons."""
    st.subheader("⚠️ Quality Rejection Reasons & Root Causes")
    st.caption("Distribution of conditions triggering 'Denied' (HIGH RISK) outcomes in inspected bunches.")

    denied_df = df[df["decision"] == "Denied"].copy()

    if denied_df.empty:
        st.success("🎉 No bunches have been denied in the selected time period! 100% Quality Pass.")
        return

    # Categorize reasons
    reason_counts = denied_df["dominant_problem"].value_counts().reset_index()
    reason_counts.columns = ["Root_Cause", "Count"]

    col1, col2 = st.columns([3, 2])

    with col1:
        reason_chart = (
            alt.Chart(reason_counts)
            .mark_bar(color=COLOR_DENIED, cornerRadius=4)
            .encode(
                x=alt.X("Count:Q", title="Denied Bunches"),
                y=alt.Y("Root_Cause:N", title="Failure Reason", sort="-x"),
                tooltip=["Root_Cause:N", "Count:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(reason_chart, use_container_width=True)

    with col2:
        st.markdown("#### Summary of Denied Batches")
        st.metric("Total Denials", len(denied_df))
        st.metric("Avg Brown % in Denied Bunches", f"{denied_df['brown_pct'].mean():.1f}%")
        st.metric("Avg Ethylene in Denied Bunches", f"{denied_df['gas_ppm'].mean():.2f} ppm")

    # Table of denied bunches
    st.markdown("---")
    st.subheader("Inspection Log of Rejected Bunches")
    display_denied = denied_df[[
        "timestamp", "bunch_id", "banana_count", "ripeness_stage",
        "brown_pct", "gas_ppm", "dominant_problem", "reasoning"
    ]].rename(columns={
        "timestamp": "Timestamp",
        "bunch_id": "Bunch ID",
        "banana_count": "Bananas",
        "ripeness_stage": "Stage",
        "brown_pct": "Brown %",
        "gas_ppm": "Ethylene (ppm)",
        "dominant_problem": "Problem",
        "reasoning": "Diagnostic Reasoning",
    })

    st.dataframe(display_denied, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: Timestamped Inspection Logs & Tables Component
# ──────────────────────────────────────────────────────────────────────────────

def _render_data_tables_tab(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """Render interactive timestamped logs and export tools."""
    st.subheader("📋 Timestamped Inspection Database")

    # Search box
    search_query = st.text_input("🔍 Search by Bunch ID or Reason", "")

    display_df = df_filtered.copy()
    if search_query:
        mask = (
            display_df["bunch_id"].astype(str).str.contains(search_query, case=False)
            | display_df["dominant_problem"].astype(str).str.contains(search_query, case=False)
            | display_df["reasoning"].astype(str).str.contains(search_query, case=False)
        )
        display_df = display_df[mask]

    st.caption(f"Displaying **{len(display_df)}** of **{len(df)}** records.")

    # Formatted table columns
    table_view = display_df[[
        "timestamp", "bunch_id", "decision", "ripeness_stage", "avg_ripeness_pct",
        "banana_count", "green_pct", "yellow_pct", "brown_pct", "gas_ppm", "dominant_problem"
    ]].rename(columns={
        "timestamp": "Timestamp",
        "bunch_id": "Bunch ID / File",
        "decision": "Decision",
        "ripeness_stage": "Stage",
        "avg_ripeness_pct": "Ripeness %",
        "banana_count": "Bananas",
        "green_pct": "Green %",
        "yellow_pct": "Yellow %",
        "brown_pct": "Brown %",
        "gas_ppm": "Gas (ppm)",
        "dominant_problem": "Observation",
    })

    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Decision": st.column_config.TextColumn(
                "Decision",
                help="Approved or Denied quality verdict",
            ),
            "Ripeness %": st.column_config.ProgressColumn(
                "Ripeness %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    # ── Daily Aggregation Summary Table ────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Daily Aggregated Quality Summary")

    if not df.empty:
        df_daily = df.copy()
        df_daily["Date"] = df_daily["timestamp"].dt.strftime("%Y-%m-%d")

        daily_agg = df_daily.groupby("Date").agg(
            Total_Bunches=("bunch_id", "count"),
            Approved=("decision", lambda s: (s == "Approved").sum()),
            Denied=("decision", lambda s: (s == "Denied").sum()),
            Avg_Ripeness=("avg_ripeness_pct", "mean"),
            Avg_Bananas=("banana_count", "mean"),
            Total_Bananas=("banana_count", "sum"),
        ).reset_index()

        daily_agg["Approval_Rate"] = (
            (daily_agg["Approved"] / daily_agg["Total_Bunches"] * 100).round(1)
        )
        daily_agg["Avg_Ripeness"] = daily_agg["Avg_Ripeness"].round(1)
        daily_agg["Avg_Bananas"] = daily_agg["Avg_Bananas"].round(1)
        daily_agg = daily_agg.sort_values(by="Date", ascending=False)

        st.dataframe(
            daily_agg.rename(columns={
                "Total_Bunches": "Total Bunches",
                "Approval_Rate": "Approval Rate (%)",
                "Avg_Ripeness": "Avg Ripeness (%)",
                "Avg_Bananas": "Avg Bananas / Bunch",
                "Total_Bananas": "Total Bananas",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── CSV Export ─────────────────────────────────────────────────────────────
    st.markdown("---")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Full 30-Day Inspection Log (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"banana_quality_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper for live simulation testing
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_live_inspection():
    """Simulate logging a real-time inspection event."""
    rng = np.random.RandomState()
    dice = rng.rand()

    if dice < 0.65:
        status = "ACCEPTABLE"
        yellow = rng.uniform(70.0, 92.0)
        green = rng.uniform(3.0, 15.0)
        brown = rng.uniform(1.0, 10.0)
        problem = "No significant problems"
        reasoning = "Ripe yellow bunch - optimal for market."
        color_signal = "ripe"
    elif dice < 0.85:
        status = "SEMI-RIPE"
        yellow = rng.uniform(40.0, 60.0)
        green = rng.uniform(25.0, 45.0)
        brown = rng.uniform(1.0, 8.0)
        problem = "Mid-ripening - green fading to yellow"
        reasoning = "Bananas mid-ripening - approved for storage."
        color_signal = "semi-ripe"
    else:
        status = "HIGH RISK"
        yellow = rng.uniform(40.0, 60.0)
        green = rng.uniform(0.0, 10.0)
        brown = rng.uniform(25.0, 45.0)
        problem = f"High brown spotting ({brown:.1f}%)"
        reasoning = "Overripe bananas present - approaching spoilage."
        color_signal = "overripe"

    banana_count = rng.randint(5, 14)
    bunch_id = f"live_bunch_{datetime.now().strftime('%H%M%S')}"

    log_inspection(
        bunch_id=bunch_id,
        banana_count=banana_count,
        avg_green=green,
        avg_yellow=yellow,
        avg_brown=brown,
        bunch_status=status,
        confidence="HIGH",
        dominant_problem=problem,
        reasoning=reasoning,
        gas_ppm=round(float(rng.uniform(0.4, 2.5)), 2),
        color_signal=color_signal,
    )
