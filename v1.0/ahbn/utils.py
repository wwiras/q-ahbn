from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime
import re

import pandas as pd


@dataclass
class ResultRow:
    experiment: str
    strategy: str
    seed: int
    num_nodes: int
    topology_type: str
    topology_param: float | int | None
    fanout: int | None
    num_clusters: int | None
    ch_overload_factor: float | None
    delivery_ratio: float
    propagation_delay: float | None
    duplicates: int
    total_forwards: int


@dataclass
class AdaptiveTraceRow:
    experiment: str
    strategy: str
    seed: int
    scenario_tag: str
    time: float
    node_id: int
    message_id: Optional[str]
    event_type: str
    mode: str
    fanout: int
    weight: float
    tau: float
    dup_ratio_raw: float
    d_hat: float
    l_hat: float
    rho_hat: float
    u_hat: float
    deg_hat: float
    ov_hat: float
    r_hat: float
    c_hat: float
    resource_class: str
    capacity_score: float
    processing_delay: float
    received_new: int
    received_duplicate: int
    forwarded: int


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_csv(
    rows: Iterable[ResultRow],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:
    df = pd.DataFrame([asdict(r) for r in rows])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if add_timestamp:
        ts = current_timestamp()
        output_path = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")

    df.to_csv(output_path, index=False)
    return str(output_path)


def save_adaptive_trace_csv(
    rows: Iterable[AdaptiveTraceRow],
    output_path: str | Path,
    add_timestamp: bool = True,
) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError("No adaptive trace rows to save.")

    df = pd.DataFrame([asdict(r) for r in rows])

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if add_timestamp:
        ts = current_timestamp()
        output_path = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")

    df.to_csv(output_path, index=False)
    return str(output_path)


def extract_timestamp_from_filename(path: str | Path) -> str | None:
    name = Path(path).name
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else None