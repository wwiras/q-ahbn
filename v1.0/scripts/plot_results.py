from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from ahbn.utils import ensure_dir, extract_timestamp_from_filename


# -----------------------------
# Helpers
# -----------------------------
def get_timestamp(csv_path: str) -> str:
    ts = extract_timestamp_from_filename(csv_path)
    if ts is None:
        raise ValueError(f"Cannot extract timestamp from {csv_path}")
    return ts


def apply_offset(series: pd.Series, offset: float) -> pd.Series:
    return series.astype(float) + offset


def get_plot_output_path(experiment: str, timestamp: str) -> str:
    return f"outputs/plots/{experiment}_combined_{timestamp}.png"


def get_exp07_3panel_output_path(timestamp: str) -> str:
    return f"outputs/plots/exp07_3panel_{timestamp}.png"


def get_adaptive_plot_output_path(experiment: str, timestamp: str) -> str:
    return f"outputs/plots/{experiment}_adaptive_{timestamp}.png"


def make_time_bins(df: pd.DataFrame, bin_width: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    out["time_bin"] = (out["time"] / bin_width).round().astype(float) * bin_width
    return out


def mode_fraction_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    mode_counts = (
        df.groupby(["time_bin", "mode"])["node_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "gossip" not in mode_counts.columns:
        mode_counts["gossip"] = 0
    if "cluster" not in mode_counts.columns:
        mode_counts["cluster"] = 0

    total = mode_counts["gossip"] + mode_counts["cluster"]
    total = total.replace(0, 1)

    mode_counts["gossip_frac"] = mode_counts["gossip"] / total
    mode_counts["cluster_frac"] = mode_counts["cluster"] / total
    return mode_counts.sort_values("time_bin")


# -----------------------------
# Exp07
# -----------------------------
def plot_exp07(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    required_cols = {
        "experiment",
        "strategy",
        "fanout",
        "delivery_ratio",
        "propagation_delay",
        "duplicates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    exp_values = set(df["experiment"].dropna().unique())
    if exp_values != {"exp07"} and "exp07" not in exp_values:
        print(
            f"Warning: CSV contains experiment values {sorted(exp_values)}. "
            f"This plotting script is intended for exp07."
        )

    df_compare = df[df["strategy"].isin(["gossip", "ahbn"])].copy()

    grouped_compare = (
        df_compare.groupby(["strategy", "fanout"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            dup_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    strategies = ["gossip", "ahbn"]
    strategies = [s for s in strategies if s in grouped_compare["strategy"].unique()]

    offsets = (
        {"gossip": -0.05, "ahbn": 0.05}
        if use_offset
        else {s: 0.0 for s in strategies}
    )

    x_ticks = sorted(df_compare["fanout"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for s in strategies:
        part = grouped_compare[grouped_compare["strategy"] == s].sort_values("fanout")
        x = apply_offset(part["fanout"], offsets[s])
        axes[0].plot(x, part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("Fanout")
    axes[0].set_ylabel("Propagation Delay")
    axes[0].set_title("Delay vs Fanout")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    for s in strategies:
        part = grouped_compare[grouped_compare["strategy"] == s].sort_values("fanout")
        x = apply_offset(part["fanout"], offsets[s])
        axes[1].plot(x, part["dup_mean"], marker="o", label=s)

    axes[1].set_xlabel("Fanout")
    axes[1].set_ylabel("Duplicates")
    axes[1].set_title("Duplicates vs Fanout")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out_combined = get_plot_output_path("exp07", ts)
    plt.savefig(out_combined, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_combined}")

    grouped_full = (
        df.groupby(["fanout", "strategy"], as_index=False)[
            ["delivery_ratio", "propagation_delay", "duplicates"]
        ]
        .mean()
        .sort_values(["fanout", "strategy"])
    )

    preferred_order = ["ahbn", "gossip", "hybrid_fixed", "cluster"]
    full_strategies = [s for s in preferred_order if s in grouped_full["strategy"].unique()]
    for s in grouped_full["strategy"].unique():
        if s not in full_strategies:
            full_strategies.append(s)

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes3[0]
    for strategy in full_strategies:
        subset = grouped_full[grouped_full["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["delivery_ratio"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Delivery Ratio vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Delivery Ratio")
    ax.grid(True, linestyle=":")
    ax.legend()

    ax = axes3[1]
    for strategy in full_strategies:
        subset = grouped_full[grouped_full["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["propagation_delay"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Delay vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Propagation Delay")
    ax.grid(True, linestyle=":")
    ax.legend()

    ax = axes3[2]
    for strategy in full_strategies:
        subset = grouped_full[grouped_full["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["duplicates"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Duplicates vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Duplicates")
    ax.grid(True, linestyle=":")
    ax.legend()

    fig3.tight_layout()
    out_3panel = get_exp07_3panel_output_path(ts)
    fig3.savefig(out_3panel, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved {out_3panel}")


# -----------------------------
# Exp08
# -----------------------------
def plot_exp08(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    df = df[df["strategy"].isin(["cluster", "ahbn"])].copy()

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    offsets = {"cluster": -0.03, "ahbn": 0.03} if use_offset else {s: 0.0 for s in strategies}

    x_ticks = sorted(df["ch_overload_factor"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[0].plot(x, part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("CH Overload Factor")
    axes[0].set_ylabel("Delay")
    axes[0].set_title("Delay vs CH Overload")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[1].plot(x, part["delivery_mean"], marker="o", label=s)

    axes[1].set_xlabel("CH Overload Factor")
    axes[1].set_ylabel("Delivery Ratio")
    axes[1].set_title("Delivery vs CH Overload")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp08", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp09
# -----------------------------
def detect_xlabel(df: pd.DataFrame) -> str:
    topo = df["topology_type"].iloc[0]
    if topo == "er":
        return "ER Edge Probability"
    elif topo == "ba":
        return "BA Attachment Parameter (m)"
    return "Topology Parameter"


def plot_exp09(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    xlabel = detect_xlabel(df)

    grouped = (
        df.groupby(["strategy", "topology_param"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            dup_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()

    if use_offset:
        offsets = {"gossip": -0.002, "cluster": 0.0, "ahbn": 0.002}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_ticks = sorted(df["topology_param"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets.get(s, 0.0))
        axes[0].plot(x, part["dup_mean"], marker="o", label=s)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Duplicates")
    axes[0].set_title("Duplicates vs Topology")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets.get(s, 0.0))
        axes[1].plot(x, part["delay_mean"], marker="o", label=s)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Delay")
    axes[1].set_title("Delay vs Topology")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp09", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp10
# -----------------------------
def plot_exp10(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "failure_mode"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
            dup_mean=("duplicates", "mean"),
            recovery_mean=("recovery_time", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    failure_modes = list(df["failure_mode"].dropna().unique())

    if use_offset:
        offsets = {"gossip": -0.06, "cluster": 0.0, "ahbn": 0.06}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_pos = list(range(len(failure_modes)))
    x_map = {mode: idx for idx, mode in enumerate(failure_modes)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 0].plot(part["x"], part["delay_mean"], marker="o", label=s)
    axes[0, 0].set_title("Delay vs Failure Mode")
    axes[0, 0].set_ylabel("Propagation Delay")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(failure_modes)
    axes[0, 0].grid(True, linestyle=":")
    axes[0, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 1].plot(part["x"], part["delivery_mean"], marker="o", label=s)
    axes[0, 1].set_title("Delivery Ratio vs Failure Mode")
    axes[0, 1].set_ylabel("Delivery Ratio")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(failure_modes)
    axes[0, 1].grid(True, linestyle=":")
    axes[0, 1].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[1, 0].plot(part["x"], part["dup_mean"], marker="o", label=s)
    axes[1, 0].set_title("Duplicates vs Failure Mode")
    axes[1, 0].set_ylabel("Duplicates")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(failure_modes)
    axes[1, 0].grid(True, linestyle=":")
    axes[1, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["failure_mode"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[1, 1].plot(part["x"], part["recovery_mean"], marker="o", label=s)
    axes[1, 1].set_title("Recovery Time vs Failure Mode")
    axes[1, 1].set_ylabel("Recovery Time")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(failure_modes)
    axes[1, 1].grid(True, linestyle=":")
    axes[1, 1].legend()

    plt.tight_layout()
    out = get_plot_output_path("exp10", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


def plot_adaptive_behavior(df: pd.DataFrame, ts: str) -> None:
    ensure_dir("outputs/plots")

    required_cols = {
        "experiment",
        "strategy",
        "time",
        "node_id",
        "fanout",
        "mode",
        "d_hat",
        "event_type",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Adaptive trace CSV is missing columns: {sorted(missing)}")

    df = df[df["strategy"] == "ahbn"].copy()
    if df.empty:
        raise ValueError("Adaptive trace CSV contains no AHBN rows.")

    df = df[df["event_type"].isin(["control_update", "forward_decision", "churn_control_update"])].copy()
    if df.empty:
        raise ValueError("No adaptive control events found after filtering.")

    df = df.sort_values(["time", "node_id"])
    df = make_time_bins(df, bin_width=0.25)

    df_last = (
        df.groupby(["time_bin", "node_id"], as_index=False)
        .last()
        .sort_values(["time_bin", "node_id"])
    )

    fanout_df = (
        df_last.groupby("time_bin", as_index=False)
        .agg(
            mean_fanout=("fanout", "mean"),
            fanout_min=("fanout", "min"),
            fanout_max=("fanout", "max"),
        )
        .sort_values("time_bin")
    )

    dup_df = (
        df_last.groupby("time_bin", as_index=False)
        .agg(
            mean_dhat=("d_hat", "mean"),
            dmin=("d_hat", "min"),
            dmax=("d_hat", "max"),
        )
        .sort_values("time_bin")
    )

    churn_df = (
        df_last.groupby("time_bin", as_index=False)
        .agg(
            mean_rhohat=("rho_hat", "mean"),
            rhomin=("rho_hat", "min"),
            rhomax=("rho_hat", "max"),
        )
        .sort_values("time_bin")
    )

    mode_df = mode_fraction_by_bin(df_last)

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.5), sharex=True)

    axes[0].plot(
        fanout_df["time_bin"],
        fanout_df["mean_fanout"],
        marker="o",
        linewidth=1.8,
        label="mean fanout",
    )
    axes[0].fill_between(
        fanout_df["time_bin"],
        fanout_df["fanout_min"],
        fanout_df["fanout_max"],
        alpha=0.20,
    )
    axes[0].set_ylabel("Mean Fanout")
    axes[0].set_title("Adaptive Fanout Over Time")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].plot(
        dup_df["time_bin"],
        dup_df["mean_dhat"],
        marker="o",
        linewidth=1.8,
        label="mean $\\hat{d}$",
    )
    axes[1].fill_between(
        dup_df["time_bin"],
        dup_df["dmin"],
        dup_df["dmax"],
        alpha=0.20,
    )
    axes[1].set_ylabel("Mean $\\hat{d}$")
    axes[1].set_title("Duplication Dynamics Over Time")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    axes[2].plot(
        churn_df["time_bin"],
        churn_df["mean_rhohat"],
        marker="o",
        linewidth=1.8,
        label="mean $\\hat{\\rho}$",
    )
    axes[2].fill_between(
        churn_df["time_bin"],
        churn_df["rhomin"],
        churn_df["rhomax"],
        alpha=0.20,
    )
    axes[2].set_ylabel("Mean $\\hat{\\rho}$")
    axes[2].set_title("Churn Perception Over Time")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    axes[3].stackplot(
        mode_df["time_bin"],
        mode_df["gossip_frac"],
        mode_df["cluster_frac"],
        labels=["gossip", "cluster"],
        alpha=0.85,
    )
    axes[3].set_xlabel("Simulation Time")
    axes[3].set_ylabel("Mode Fraction")
    axes[3].set_title("Dissemination Mode Usage Over Time")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].grid(True, linestyle=":")
    axes[3].legend(loc="upper right")

    # --- ADD THIS BLOCK ---
    FAILURE_TIME = 1.0  # match your config trigger_time

    for ax in axes:
        ax.axvline(
            x=FAILURE_TIME,
            linestyle="--",
            color="red",
            alpha=0.7,
            linewidth=1.5,
            label="failure event"
        )

    # Optional: show legend only once (top plot)
    axes[0].legend(loc="upper right")
    # --- END PATCH ---

    plt.tight_layout()
    experiment = df["experiment"].iloc[0]
    out = get_adaptive_plot_output_path(experiment, ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp11
# -----------------------------
def plot_exp11(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "churn_rate"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
            dup_mean=("duplicates", "mean"),
            adapt_mean=("adaptation_rate", "mean"),
        )
        .reset_index()
    )

    strategies = [s for s in ["gossip", "cluster", "ahbn"] if s in grouped["strategy"].unique()]

    if use_offset:
        offsets = {"gossip": -0.006, "cluster": 0.0, "ahbn": 0.006}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_ticks = sorted(df["churn_rate"].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("churn_rate")
        x = apply_offset(part["churn_rate"], offsets.get(s, 0.0))
        axes[0, 0].plot(x, part["delay_mean"], marker="o", label=s)
    axes[0, 0].set_title("Delay vs Churn Rate")
    axes[0, 0].set_ylabel("Propagation Delay")
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].grid(True, linestyle=":")
    axes[0, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("churn_rate")
        x = apply_offset(part["churn_rate"], offsets.get(s, 0.0))
        axes[0, 1].plot(x, part["delivery_mean"], marker="o", label=s)
    axes[0, 1].set_title("Delivery Ratio vs Churn Rate")
    axes[0, 1].set_ylabel("Delivery Ratio")
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].grid(True, linestyle=":")
    axes[0, 1].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("churn_rate")
        x = apply_offset(part["churn_rate"], offsets.get(s, 0.0))
        axes[1, 0].plot(x, part["dup_mean"], marker="o", label=s)
    axes[1, 0].set_title("Duplicates vs Churn Rate")
    axes[1, 0].set_ylabel("Duplicates")
    axes[1, 0].set_xticks(x_ticks)
    axes[1, 0].grid(True, linestyle=":")
    axes[1, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("churn_rate")
        x = apply_offset(part["churn_rate"], offsets.get(s, 0.0))
        axes[1, 1].plot(x, part["adapt_mean"], marker="o", label=s)
    axes[1, 1].set_title("Adaptation Rate vs Churn Rate")
    axes[1, 1].set_ylabel("Adaptation Rate")
    axes[1, 1].set_xticks(x_ticks)
    axes[1, 1].grid(True, linestyle=":")
    axes[1, 1].legend()

    for ax in axes[1, :]:
        ax.set_xlabel("Churn Rate")

    plt.tight_layout()
    out = get_plot_output_path("exp11", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp12
# -----------------------------
def plot_exp12(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "resource_scenario"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
            dup_mean=("duplicates", "mean"),
            load_cv_mean=("load_balance_cv", "mean"),
        )
        .reset_index()
    )

    strategies = [s for s in ["gossip", "cluster", "ahbn"] if s in grouped["strategy"].unique()]
    scenarios = list(df["resource_scenario"].dropna().unique())

    if use_offset:
        offsets = {"gossip": -0.08, "cluster": 0.0, "ahbn": 0.08}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_pos = list(range(len(scenarios)))
    x_map = {sc: idx for idx, sc in enumerate(scenarios)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["resource_scenario"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 0].plot(part["x"], part["delay_mean"], marker="o", label=s)
    axes[0, 0].set_title("Delay vs Resource Scenario")
    axes[0, 0].set_ylabel("Propagation Delay")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(scenarios)
    axes[0, 0].grid(True, linestyle=":")
    axes[0, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["resource_scenario"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[0, 1].plot(part["x"], part["delivery_mean"], marker="o", label=s)
    axes[0, 1].set_title("Delivery Ratio vs Resource Scenario")
    axes[0, 1].set_ylabel("Delivery Ratio")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(scenarios)
    axes[0, 1].grid(True, linestyle=":")
    axes[0, 1].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["resource_scenario"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[1, 0].plot(part["x"], part["dup_mean"], marker="o", label=s)
    axes[1, 0].set_title("Duplicates vs Resource Scenario")
    axes[1, 0].set_ylabel("Duplicates")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(scenarios)
    axes[1, 0].grid(True, linestyle=":")
    axes[1, 0].legend()

    for s in strategies:
        part = grouped[grouped["strategy"] == s].copy()
        part["x"] = part["resource_scenario"].map(x_map).astype(float) + offsets.get(s, 0.0)
        part = part.sort_values("x")
        axes[1, 1].plot(part["x"], part["load_cv_mean"], marker="o", label=s)
    axes[1, 1].set_title("Load Balance CV vs Resource Scenario")
    axes[1, 1].set_ylabel("Load Balance CV")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scenarios)
    axes[1, 1].grid(True, linestyle=":")
    axes[1, 1].legend()

    plt.tight_layout()
    out = get_plot_output_path("exp12", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_adaptive_behavior_exp12(df: pd.DataFrame, ts: str) -> None:
    ensure_dir("outputs/plots")

    df = df[df["strategy"] == "ahbn"].copy()
    if df.empty:
        raise ValueError("Adaptive trace CSV contains no AHBN rows.")

    df = df[df["event_type"].isin(["control_update", "forward_decision"])].copy()
    if df.empty:
        raise ValueError("No adaptive control events found after filtering.")

    df = df.sort_values(["time", "node_id"])
    df = make_time_bins(df, bin_width=0.25)
    df_last = df.groupby(["time_bin", "node_id"], as_index=False).last()

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.5), sharex=True)

    for cls_name in ["strong", "medium", "weak"]:
        part = (
            df_last[df_last["resource_class"] == cls_name]
            .groupby("time_bin", as_index=False)
            .agg(mean_fanout=("fanout", "mean"), mean_weight=("weight", "mean"), mean_chat=("c_hat", "mean"))
        )
        if part.empty:
            continue
        axes[0].plot(part["time_bin"], part["mean_fanout"], marker="o", label=cls_name)
        axes[1].plot(part["time_bin"], part["mean_weight"], marker="o", label=cls_name)
        axes[2].plot(part["time_bin"], part["mean_chat"], marker="o", label=cls_name)

    load_df = df_last.copy()
    load_df["norm_forwarded"] = load_df["forwarded"] / load_df["capacity_score"].clip(lower=0.25)
    for cls_name in ["strong", "medium", "weak"]:
        part = (
            load_df[load_df["resource_class"] == cls_name]
            .groupby("time_bin", as_index=False)
            .agg(mean_norm_forwarded=("norm_forwarded", "mean"))
        )
        if part.empty:
            continue
        axes[3].plot(part["time_bin"], part["mean_norm_forwarded"], marker="o", label=cls_name)

    axes[0].set_ylabel("Mean Fanout")
    axes[0].set_title("Class-wise Fanout Over Time")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].set_ylabel("Mean Weight")
    axes[1].set_title("Class-wise Gossip Preference Over Time")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    # axes[2].set_ylabel("Mean $\hat{c}$")
    axes[2].set_ylabel("Mean $\\hat{c}$")
    axes[2].set_title("Capacity Stress Perception Over Time")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    axes[3].set_xlabel("Simulation Time")
    axes[3].set_ylabel("Norm. Forwarded")
    axes[3].set_title("Normalized Forwarding Load Over Time")
    axes[3].grid(True, linestyle=":")
    axes[3].legend()

    plt.tight_layout()
    out = get_adaptive_plot_output_path("exp12", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# -----------------------------
# Main
# -----------------------------
def main(path: str, use_offset: bool) -> None:
    df = pd.read_csv(path)
    ts = get_timestamp(path)

    if "experiment" not in df.columns:
        raise ValueError("CSV must contain an 'experiment' column")

    adaptive_trace_cols = {"time", "node_id", "fanout", "mode", "d_hat"}
    if adaptive_trace_cols.issubset(df.columns):
        experiment = df["experiment"].iloc[0]
        if experiment == "exp12" and "resource_class" in df.columns and "c_hat" in df.columns:
            plot_adaptive_behavior_exp12(df, ts)
        else:
            plot_adaptive_behavior(df, ts)
        print(f"Adaptive plots saved with timestamp: {ts}")
        return

    experiment = df["experiment"].iloc[0]
    df = df[df["experiment"] == experiment].copy()

    if experiment == "exp07":
        plot_exp07(df, ts, use_offset)
    elif experiment == "exp08":
        plot_exp08(df, ts, use_offset)
    elif experiment == "exp09":
        plot_exp09(df, ts, use_offset)
    elif experiment == "exp10":
        plot_exp10(df, ts, use_offset=use_offset)
    elif experiment == "exp11":
        plot_exp11(df, ts, use_offset=use_offset)
    elif experiment == "exp12":
        plot_exp12(df, ts, use_offset=use_offset)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print(f"Plots saved (offset={use_offset}) with timestamp: {ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument(
        "--offset",
        action="store_true",
        help="Enable slight x-offset to separate overlapping curves",
    )
    args = parser.parse_args()

    main(args.csv_path, args.offset)