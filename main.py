#!/usr/bin/env python3
"""Final TrajFlow/PPO command line entry point.

This file intentionally keeps only the current workflow:

  * precompute: TrajFlow -> per-scenario heatmaps
  * train:      multi-scenario PPO with acceleration/heading-rate control
  * viz:        single-scenario rollout visualization
  * report:     plot multi-scenario training history

Each subcommand accepts a small stable set of convenience options here, and
forwards any extra flags to the underlying script.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
FINAL_POLICY = "runs/ppo_final.pt"
FINAL_HISTORY = "runs/ppo_final.json"
FINAL_REPORT = "runs/ppo_final_report.png"
FINAL_TRAJFLOW = "trajflow_GRU_DNF_marginal_AV2_best_all.pt"


def _has_flag(args: Sequence[str], *names: str) -> bool:
    prefixes = tuple(name + "=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in args)


def _inject_default(args: List[str], names: Iterable[str], *value: str) -> None:
    names = tuple(names)
    if not _has_flag(args, *names):
        args.extend([names[0], *value])


def _inject_bool_default(args: List[str], positive: str, negative: str, enabled: bool) -> None:
    if _has_flag(args, positive, negative):
        return
    args.append(positive if enabled else negative)


def _append_goal_args(args: List[str], ns: argparse.Namespace) -> None:
    if _has_flag(args, "--goal_from_av_endpoint", "--goal_world_x", "--goal_world_y"):
        return
    if ns.goal == "none":
        return
    if ns.goal == "endpoint":
        args.extend(["--goal_from_av_endpoint", "--goal_lane_offset_m", str(ns.goal_lane_offset_m)])
        return
    if ns.goal == "world":
        if ns.goal_world_x is None or ns.goal_world_y is None:
            raise SystemExit("--goal world requires --goal_world_x and --goal_world_y")
        args.extend(["--goal_world_x", str(ns.goal_world_x), "--goal_world_y", str(ns.goal_world_y)])


def _run_module(module_name: str, argv: Sequence[str]) -> None:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(REPO_ROOT / (module_name.replace(".", "/") + ".py")), *argv]
        module.main()
    finally:
        sys.argv = old_argv


def _train(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="main.py train",
        description="Train the final multi-scenario PPO policy. Unknown flags are forwarded.",
    )
    parser.add_argument("--goal", choices=("endpoint", "none", "world"), default="endpoint")
    parser.add_argument("--goal_lane_offset_m", type=float, default=0.0)
    parser.add_argument("--goal_world_x", type=float, default=None)
    parser.add_argument("--goal_world_y", type=float, default=None)
    ns, rest = parser.parse_known_args(argv)

    out = list(rest)
    _append_goal_args(out, ns)

    # Defaults from the final multi-scenario TTC/collision run. Pass any of
    # these flags explicitly to override them.
    _inject_default(out, ("--save_policy",), FINAL_POLICY)
    _inject_default(out, ("--save_history",), FINAL_HISTORY)
    _inject_default(out, ("--total_updates",), "100")
    _inject_default(out, ("--rollout_steps",), "1024")
    _inject_default(out, ("--value_coef",), "0.02")
    _inject_default(out, ("--entropy_coef",), "0.005")
    _inject_default(out, ("--log_std_init",), "-0.7")
    _inject_default(out, ("--w_forward_progress",), "0.3")
    _inject_default(out, ("--w_off_map",), "3.0")
    _inject_default(out, ("--w_back",), "3.0")
    _inject_default(out, ("--w_lane_lateral",), "0.5")
    _inject_default(out, ("--progress_cap_m",), "1.5")
    _inject_default(out, ("--reward_clip",), "10.0")
    _inject_default(out, ("--invalid_penalty",), "0.0")
    _inject_default(out, ("--heading_rate_max_radps",), "0.3")
    _inject_default(out, ("--ttc_conflict_penalty",), "1.0")
    _inject_default(out, ("--collision_penalty",), "30.0")
    _inject_bool_default(out, "--gate_progress_on_map", "--no_gate_progress_on_map", True)
    _inject_bool_default(out, "--terminate_on_invalid", "--no_terminate_on_invalid", False)
    _inject_bool_default(out, "--dynamic_forward_unit", "--no_dynamic_forward_unit", True)
    if not _has_flag(out, "--collision_penalty_once"):
        out.append("--collision_penalty_once")
    if not _has_flag(out, "--terminate_on_collision"):
        out.append("--terminate_on_collision")

    _run_module("scripts.train_ppo_multi_scenario_accel", out)


def _viz(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="main.py viz",
        description="Visualize one scenario with the final control-mode policy. Unknown flags are forwarded.",
    )
    parser.add_argument("--goal", choices=("endpoint", "none", "world"), default="endpoint")
    parser.add_argument("--goal_lane_offset_m", type=float, default=0.0)
    parser.add_argument("--goal_world_x", type=float, default=None)
    parser.add_argument("--goal_world_y", type=float, default=None)
    ns, rest = parser.parse_known_args(argv)

    out = list(rest)
    _append_goal_args(out, ns)
    _inject_default(out, ("--policy",), FINAL_POLICY)
    _inject_default(out, ("--out",), "runs/viz_one/ppo_final.png")
    _inject_default(out, ("--w_forward_progress",), "0.3")
    _inject_default(out, ("--w_off_map",), "3.0")
    _inject_default(out, ("--w_back",), "3.0")
    _inject_default(out, ("--w_lane_lateral",), "0.5")
    _inject_default(out, ("--progress_cap_m",), "1.5")
    _inject_default(out, ("--reward_clip",), "10.0")
    _inject_default(out, ("--invalid_penalty",), "0.0")
    _inject_default(out, ("--heading_rate_max_radps",), "0.3")
    _inject_bool_default(out, "--gate_progress_on_map", "--no_gate_progress_on_map", True)
    _inject_bool_default(out, "--terminate_on_invalid", "--no_terminate_on_invalid", False)
    _inject_bool_default(out, "--dynamic_forward_unit", "--no_dynamic_forward_unit", True)
    if not _has_flag(out, "--control_mode"):
        out.append("--control_mode")

    _run_module("scripts.viz_one_scenario", out)


def _precompute(argv: Sequence[str]) -> None:
    if _has_flag(argv, "-h", "--help"):
        print(
            "usage: python main.py precompute --scenarios PATH [options]\n\n"
            "Defaults: --model trajflow_GRU_DNF_marginal_AV2_best_all.pt --encoder GRU --flow DNF\n"
            "Common options forwarded to scripts/precompute_scenario_heatmaps.py:\n"
            "  --out data/heatmaps_rl_s100 --steps 64 --device cuda --only_scenarios N --overwrite"
        )
        return
    out = list(argv)
    _inject_default(out, ("--model",), FINAL_TRAJFLOW)
    _inject_default(out, ("--flow",), "DNF")
    _inject_default(out, ("--encoder",), "GRU")
    _run_module("scripts.precompute_scenario_heatmaps", out)


def _report(argv: Sequence[str]) -> None:
    if _has_flag(argv, "-h", "--help"):
        print(
            "usage: python main.py report [--history runs/ppo_final.json] [--out runs/ppo_final_report.png]\n\n"
            "Extra flags are forwarded to scripts/plot_multi_scenario_ppo_report.py."
        )
        return
    out = list(argv)
    _inject_default(out, ("--history",), FINAL_HISTORY)
    _inject_default(out, ("--out",), FINAL_REPORT)
    _run_module("scripts.plot_multi_scenario_ppo_report", out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python main.py {precompute,train,viz,report} [options]\n\n"
            "examples:\n"
            "  python main.py train --goal endpoint --goal_lane_offset_m 2.0\n"
            "  python main.py train --goal none --save_policy runs/ppo_no_goal.pt\n"
            "  python main.py viz --policy runs/ppo_final.pt --scenario_id <id> --out runs/viz_one/final.gif\n"
            "  python main.py precompute --scenarios data/av2_mf_tiny/scenarios_rl_moving_av_100.pt\n"
        )
        return

    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        _train(rest)
    elif cmd == "viz":
        _viz(rest)
    elif cmd == "precompute":
        _precompute(rest)
    elif cmd == "report":
        _report(rest)
    else:
        raise SystemExit(f"unknown command {cmd!r}; expected precompute, train, viz, or report")


if __name__ == "__main__":
    main()
