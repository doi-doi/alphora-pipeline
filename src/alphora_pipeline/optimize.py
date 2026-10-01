"""Training eligibility and connected Sharpe plateaus, isolated by model family."""

import json
from collections import deque

import numpy as np


def key(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def neighbors(parameters, grid):
    for axis, values in grid.items():
        if axis == "model":
            continue
        ordered = sorted(values)
        index = ordered.index(parameters[axis])
        for n in (index - 1, index + 1):
            if 0 <= n < len(ordered):
                yield {**parameters, axis: ordered[n]}


def distance(a, b, grid):
    return sum(
        ((a[k] - b[k]) / (max(v) - min(v))) ** 2 for k, v in grid.items() if k != "model" and max(v) != min(v)
    )


def select(records, cfg):
    eligible = [r for r in records if r["eligible"] and r["sharpe"] is not None]
    if not eligible:
        return None
    best = max(r["sharpe"] for r in eligible)
    tolerance = max(
        cfg["optimization"]["absolute_tolerance"], abs(best) * cfg["optimization"]["relative_tolerance"]
    )
    plateau = [r for r in eligible if r["sharpe"] >= best - tolerance]
    lookup = {key(r["parameters"]): r for r in plateau}
    if cfg["optimization"]["selection"] == "legacy_grid_center":
        grid = cfg["grid"]
        middle = {k: (max(v) + min(v)) / 2 for k, v in grid.items() if k != "model"}
        chosen = min(
            plateau,
            key=lambda r: (distance(r["parameters"], middle, grid), -r["sharpe"], key(r["parameters"])),
        )
        return {
            **chosen,
            "plateau_size": len(plateau),
            "best_sharpe": best,
            "tolerance": tolerance,
            "selection": "legacy_grid_center",
        }
    # Numeric adjacency is meaningful within a model; categorical model labels are not distances.
    remaining = set(lookup)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component, queue = [], deque([seed])
        while queue:
            current = queue.popleft()
            component.append(lookup[current])
            for p in neighbors(lookup[current]["parameters"], cfg["grid"]):
                candidate = key(p)
                if candidate in remaining:
                    remaining.remove(candidate)
                    queue.append(candidate)
        components.append(component)
    component = min(
        components,
        key=lambda c: (-len(c), -float(np.median([r["sharpe"] for r in c])), key(c[0]["parameters"])),
    )
    # Closest discrete point to the component centroid; no preference for an unrelated global center.
    center = {k: float(np.mean([r["parameters"][k] for r in component])) for k in cfg["grid"] if k != "model"}
    chosen = min(
        component,
        key=lambda r: (distance(r["parameters"], center, cfg["grid"]), -r["sharpe"], key(r["parameters"])),
    )
    return {
        **chosen,
        "plateau_size": len(component),
        "near_best_size": len(plateau),
        "best_sharpe": best,
        "tolerance": tolerance,
        "selection": "connected_plateau",
        "singleton_plateau": len(component) == 1,
    }


def eligibility(metrics, valid_bars, options):
    reasons = []
    if valid_bars < options["min_valid_bars"]:
        reasons.append("insufficient_valid_signal_bars")
    if metrics["trade_events"] < options["min_trade_events"]:
        reasons.append("insufficient_trade_events")
    if metrics["sharpe"] is None:
        reasons.append("undefined_sharpe")
    return not reasons, reasons
