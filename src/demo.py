"""The 60-second artefact: perplexity says 4-bit is nearly free, and it is lying.

Run with:  python -m src.demo
"""

from __future__ import annotations

import math
import random
import statistics

from .backend import ModelBackend, default_sweep_configs
from .frontier import (
    adjacent_sizing,
    frontier_by_task,
    min_eval_size,
    profile_sweep,
    select_on_metric,
)
from .generator import make_weight_tensor
from .tinymodel import TinyMLP
from .quantise import QuantConfig, per_channel_errors, quantise_group, quantise_matrix

RULE = "=" * 78
THIN = "-" * 78

# Quality bar used for the "pick a config" demonstrations. 0.80 on the 0-1
# normalised scale is a deliberately LENIENT bar -- the point is that even a
# lenient bar on the aggregate metric admits a config with a dead capability.
QUALITY_BAR = 0.80
# Bar for the per-task recommendation. A structured-output endpoint that emits
# invalid JSON 5% of the time is already a production incident.
TASK_BAR = 0.95
# A second, deliberately tolerant bar. At 0.95 every task happens to land on
# the same config; at 0.75 they separate, which is the point -- "which config"
# is a function of the task AND of how much degradation you will accept.
TOLERANT_BAR = 0.75


def banner(title: str) -> None:
    print()
    print(RULE)
    print(title)
    print(RULE)


def section_limitation() -> None:
    banner("quantfrontier -- quality/latency frontier on DOWNSTREAM TASKS")
    print("LIMITATION, stated up front: there is no model runtime here. No weights,")
    print("no GPU, no network. The quantisation ARITHMETIC below is real and runs on")
    print("real float matrices; the mapping from weight error to task accuracy,")
    print("latency and memory is a seeded deterministic simulation. What is being")
    print("demonstrated and tested is the MEASUREMENT METHODOLOGY.")


def section_arithmetic() -> None:
    banner("1. The quantiser is real: measured error vs closed form")
    print("Uniform data on [-1, 1], per-tensor symmetric. For a uniform quantiser")
    print("the RMS reconstruction error is step/sqrt(12). Measured vs predicted:")
    print()
    rng = random.Random(7)
    values = [rng.uniform(-1.0, 1.0) for _ in range(120000)]
    print("  %-6s %-12s %-12s %-8s" % ("bits", "measured", "step/sqrt12", "ratio"))
    for bits in (8, 6, 5, 4, 3):
        cfg = QuantConfig(bits=bits, per_channel=False, symmetric=True)
        recon, scale, _ = quantise_group(values, cfg)
        err = math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(values, recon)))
        theory = scale / math.sqrt(12.0)
        print("  %-6d %-12.6f %-12.6f %-8.4f" % (bits, err, theory, err / theory))


def section_granularity() -> None:
    banner("2. Per-channel vs per-tensor, and WHY")
    tensor, planted = make_weight_tensor(96, 128, seed=1, outlier_channels=4)
    clean, _ = make_weight_tensor(96, 128, seed=1, outlier_channels=0)
    planted_set = set(planted)

    print("Same tensor, 96 channels, 4 of them planted 20x larger than the bulk.")
    print()
    print("  %-6s %-12s %-12s %-8s" % ("bits", "per-channel", "per-tensor", "ratio"))
    for bits in (8, 6, 4, 3):
        pc = statistics.fmean(per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(bits, True, True))))
        pt = statistics.fmean(per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(bits, False, True))))
        print("  %-6d %-12.4f %-12.4f %-8.2f" % (bits, pc, pt, pt / pc))

    err_pt = per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(4, False, True)))
    err_pc = per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(4, True, True)))
    bulk_pt = statistics.fmean(e for i, e in enumerate(err_pt) if i not in planted_set)
    out_pt = statistics.fmean(e for i, e in enumerate(err_pt) if i in planted_set)
    bulk_pc = statistics.fmean(e for i, e in enumerate(err_pc) if i not in planted_set)
    print()
    print("Mechanism, at 4 bits. Per-tensor picks ONE scale from the global max,")
    print("which the outlier channels set. The bulk then has nowhere to go:")
    print("  outlier channels, per-tensor : %.4f relative error" % out_pt)
    print("  bulk channels,    per-tensor : %.4f  <- rounded essentially to zero" % bulk_pt)
    print("  bulk channels,    per-channel: %.4f" % bulk_pc)
    print()

    ratio_dirty = (
        statistics.fmean(per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(4, False, True))))
        / statistics.fmean(per_channel_errors(tensor, quantise_matrix(tensor, QuantConfig(4, True, True))))
    )
    ratio_clean = (
        statistics.fmean(per_channel_errors(clean, quantise_matrix(clean, QuantConfig(4, False, True))))
        / statistics.fmean(per_channel_errors(clean, quantise_matrix(clean, QuantConfig(4, True, True))))
    )
    print("Control: the SAME tensor generated without planted outlier channels.")
    print("  per-tensor/per-channel error ratio, with outliers   : %.2fx" % ratio_dirty)
    print("  per-tensor/per-channel error ratio, without outliers: %.2fx" % ratio_clean)
    print("  outlier-attributable share of the advantage         : %.1fx" % (ratio_dirty / ratio_clean))


def section_sweep(measurements) -> None:
    banner("3. The sweep")
    tasks = list(measurements[0].task_accuracy)
    head = "  %-20s %7s %7s %7s %9s %8s" % ("config", "w_err", "ppl", "agg_q", "lat_ms", "mem_GB")
    head += "".join(" %10s" % t[:10] for t in tasks)
    print(head)
    print("  " + THIN)
    for m in measurements:
        row = "  %-20s %7.4f %7.3f %7.3f %5.1f+-%.2f %8.2f" % (
            m.config.name,
            m.weight_error,
            m.perplexity,
            m.aggregate_quality,
            m.latency_ms_mean,
            m.latency_sem,
            m.memory_mb / 1000.0,
        )
        row += "".join(" %10.3f" % m.task_accuracy[t] for t in tasks)
        print(row)
    print()
    print("  Latency is mean +- standard error over %d replications." % measurements[0].replications)


def section_cliff(ladder) -> None:
    banner("4. Smooth aggregate, cliff-edged task")
    profiles = profile_sweep(ladder)
    print("  cliff_share = largest single-step drop / total drop across the sweep.")
    print("  A perfectly smooth curve gives %.3f here; a cliff gives ~1.0." % profiles["aggregate"].smooth_share)
    print()
    print("  steps@90%% = fewest of the %d sweep steps that explain 90%% of the total"
          % (len(ladder) - 1))
    print("  degradation. 1 means a cliff; most of them means a gradual slide.")
    print()
    print("  %-22s %8s %10s %11s %8s  %s"
          % ("metric", "total", "worst step", "cliff_share", "steps@90%", "worst step at"))
    print("  " + THIN)
    for name in ["aggregate"] + [k for k in profiles if k != "aggregate"]:
        p = profiles[name]
        print(
            "  %-22s %8.3f %10.3f %11.3f %8d  %s -> %s"
            % (name, p.total_drop, p.max_step_drop, p.cliff_share, p.steps_to_share(0.90),
               p.cliff_from, p.cliff_at)
        )
    print()
    print("  Aggregate loses %.3f in its worst step. json_schema_valid loses %.3f"
          % (profiles["aggregate"].max_step_drop, profiles["json_schema_valid"].max_step_drop))
    print("  in the SAME direction of travel -- %.1fx as much, in one bit."
          % (profiles["json_schema_valid"].max_step_drop / profiles["aggregate"].max_step_drop))


def section_trap(measurements) -> None:
    banner("5. The trap: what a single aggregate number ships")
    pick = select_on_metric(measurements, lambda m: m.aggregate_quality, QUALITY_BAR)
    print("  Rule: 'smallest model whose aggregate quality stays above %.2f'" % QUALITY_BAR)
    print("  Picks: %s  (perplexity %.2f, only %.1f%% above full precision)"
          % (pick.config.name, pick.perplexity, 100.0 * (pick.perplexity / measurements[0].perplexity - 1.0)))
    print()
    print("  What that config actually does on each task:")
    for task, acc in pick.task_accuracy.items():
        flag = "  <- BROKEN" if acc < 0.5 else ""
        print("    %-24s %.3f%s" % (task, acc, flag))
    print()
    print("  Selecting per task instead, at two quality bars:")
    print("    %-24s %-26s %s" % ("task", "bar %.2f" % TASK_BAR, "bar %.2f" % TOLERANT_BAR))
    for task in pick.task_accuracy:
        cells = []
        for bar in (TASK_BAR, TOLERANT_BAR):
            safe = select_on_metric(measurements, lambda m, t=task: m.task_accuracy[t], bar)
            cells.append(
                "%s (%.2f GB)" % (safe.config.name, safe.memory_mb / 1000.0) if safe else "none clears the bar"
            )
        print("    %-24s %-26s %s" % (task, cells[0], cells[1]))
    print()
    safe = select_on_metric(measurements, lambda m: m.task_accuracy["json_schema_valid"], TASK_BAR)
    gap = safe.config.bits - pick.config.bits
    print("  Same sweep, same data. The aggregate metric and the structured-output")
    print("  task disagree by %d bit width(s)." % gap)


def section_tiny() -> None:
    banner("8. REAL forward passes: tiny MLP (16-48-10), planted outlier rows")
    print("  Not simulated: weights quantised, network run, scores measured.")
    print("  Labels = teacher argmax with a planted 5% flip rate, so the FP")
    print("  expectations are 0.950 (classification) and 0.95^8=0.663 (record).")
    print()
    model = TinyMLP()
    cfgs = [QuantConfig(b) for b in (16, 8, 6, 5, 4, 3, 2)] + [QuantConfig(4, per_channel=False)]
    rows = [model.evaluate(c, replications=3) for c in cfgs]
    fp = rows[0].task_accuracy
    print("  %-14s %9s %9s %8s %8s %8s %8s %9s" % ("config", "ppl", "KL nats", "class", "record",
                                                 "ret_cls", "ret_rec", "lat_ms"))
    print("  " + THIN)
    for m in rows:
        a = m.task_accuracy
        print("  %-14s %9.3g %9.4f %8.3f %8.3f %8.2f %8.2f %5.3f+-%.3f" % (
            m.config.name, m.perplexity, m.kl_divergence, a["classification"], a["structured_record"],
            a["classification"] / fp["classification"], a["structured_record"] / fp["structured_record"],
            m.latency_ms_mean, m.latency_sem))
    print()
    print("  ret_* = retention vs FP. Latency is wall-clock, pure Python, noisy,")
    print("  and does not depend on bit width here (dequantised floats) -- report only.")


def section_pareto(measurements) -> None:
    banner("6. Pareto frontier per task (quality up, latency down, memory down)")
    results = frontier_by_task(measurements)
    for task, res in results.items():
        print()
        print("  %s" % task)
        print("    frontier : %s" % ", ".join(res.frontier_names))
        if res.dominated:
            for loser, winner in res.dominated:
                print("    dominated: %-20s by %s" % (loser.config.name, winner.config.name))
        else:
            print("    dominated: (none)")
    print()
    sets = {t: frozenset(r.frontier_names) for t, r in results.items()}
    distinct = len(set(sets.values()))
    print("  %d distinct frontiers across %d objectives -- the best configuration"
          % (distinct, len(sets)))
    print("  is NOT the same for every task, which is the entire argument for")
    print("  reporting the frontier per task instead of once.")


def section_sizing(ladder) -> None:
    banner("7. How big does the eval set have to be?")
    print("  Two-proportion sizing, alpha=0.05, power=0.80, items per config.")
    print("  Large n means the two configs are nearly tied -- a small eval set")
    print("  cannot tell them apart and any frontier drawn from it is noise.")
    print()
    for task in ("aggregate", "summarisation_rouge", "json_schema_valid"):
        print("  %s" % task)
        for row in adjacent_sizing(ladder, task):
            n = row.n_required
            shown = ">10^6" if n > 10 ** 6 else "%d" % n
            print("    %-14s -> %-14s  %.4f -> %.4f   n >= %s"
                  % (row.lower, row.upper, row.p_lower, row.p_upper, shown))
        print()
    n50 = min_eval_size(0.90, 0.85)
    print("  Worked example: telling 0.900 from 0.850 needs n >= %d per config." % n50)
    print("  A 100-item eval set cannot do it. Most published quantisation")
    print("  comparisons are run on fewer items than that.")


def main() -> None:
    section_limitation()
    section_arithmetic()
    section_granularity()

    backend = ModelBackend()
    measurements = backend.sweep(default_sweep_configs(), replications=12)
    ladder = [m for m in measurements if m.config.per_channel and m.config.outlier_fraction == 0.0]

    section_sweep(measurements)
    section_cliff(ladder)
    section_trap(measurements)
    section_pareto(measurements)
    section_sizing(ladder)
    section_tiny()

    banner("Headline")
    lost = 1.0 - ladder[4].task_accuracy["json_schema_valid"] / ladder[0].task_accuracy["json_schema_valid"]
    print("  (simulated backend) 4-bit costs %.1f%% perplexity and %.1f%% of structured output."
          % (100.0 * (ladder[4].perplexity / ladder[0].perplexity - 1.0), 100.0 * lost))
    print("  A benchmark that reports one number recommends it.")
    print(RULE)


if __name__ == "__main__":
    main()
