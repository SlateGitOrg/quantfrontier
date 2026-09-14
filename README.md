# quantfrontier

> A quantisation quality/latency frontier measured on downstream tasks, because perplexity hides the cliff.

> **Implementation note.** No LLM, llama.cpp, GPU or model weights are used or
> required. Quantisation (int2-int16, per-tensor vs per-channel, symmetric vs
> asymmetric, fp outlier retention) is real integer arithmetic in pure Python.
> It runs on (a) a tiny 16-48-10 MLP with planted outlier rows, where accuracy,
> KL divergence and perplexity come from actual forward passes, and (b) a
> seeded simulated backend (`src/backend.py`). That backend maps *measured*
> weight error to per-task accuracy, latency and memory for a notional 7B
> deployment. None of the numbers below are real-LLM benchmark results.
> Real models plug in by implementing `forward -> logits` (for example,
> llama.cpp logits per GGUF quant) and returning the same `Measurement`
> records. The frontier, cliff and sizing code is unchanged.


`COMPACT` · **AI / ML Engineering** · Advanced · ~5-6 days · Manufacturing - on-premise inference on modest hardware

**Primary language:** Python
**Tags:** `quantisation`, `llm`, `benchmarking`, `llama-cpp`, `pareto`, `evaluation`

---

## The problem

A team needs a model to run on hardware they already own. They pick a quantisation level from a forum post, find the output subtly worse in ways perplexity does not capture, and have no framework at all for the trade they are making - so the decision gets revisited every time somebody notices a bad output.

## ⭐ The differentiator

Evaluates quantisation on **task-specific downstream metrics and output-distribution divergence, not perplexity**. Perplexity degrades smoothly and monotonically while task performance frequently falls off a cliff at a specific bit width - and the cliff is the only thing you need to find. It reports a genuine Pareto frontier of quality against latency and memory **on the actual target hardware**, per task, because the cliff sits in a different place for summarisation than for structured extraction.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

Open-weight models via Ollama / llama.cpp (Qwen, Llama, Phi - all free) and public task benchmarks via lm-eval-harness.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python
- llama.cpp / GGUF for quantisation and inference
- lm-eval-harness
- DuckDB, Quarto
- pytest

## Core capabilities

- Automated quantisation sweep across bit widths and methods
- Task-battery evaluation with per-task scoring - never a single aggregate number
- Output-distribution divergence (KL against full precision) as an early-warning signal
- Latency, memory and throughput measured locally with variance reported
- Pareto frontier report with an explicit recommendation per deployment constraint

## Repository layout

```
src/quantise/
src/eval/
src/bench/
results/
report/
test/
```

## Build plan

1. Validate the harness against published full-precision scores before trusting any delta it reports.
2. Sweep quantisation levels and measure per task. Look for the cliff, and note that it moves between tasks.
3. Add KL divergence as the cheap early-warning signal.
4. Pareto report and the per-constraint recommendation last.

## Testing strategy

Assert measurement reproducibility within a stated variance across repeated runs - latency benchmarks on a laptop are noisy and an unreplicated number is not a result. Assert **full-precision baseline scores match published values within tolerance**, validating the harness before any of its deltas are believed.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

Every reported figure carries its replication count and variance. The report refuses to draw a frontier from single-run measurements.

## Measurable outcome

> A Pareto chart where one task holds quality down to 4-bit and another collapses at 5-bit - two facts a single aggregate score would have hidden.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo`, deterministic seeds, about 7 s. The suite has 29 tests and runs in about 28 s.

**Real forward passes (tiny MLP, 1600 eval items):**

| config | ppl | KL (nats) | classification | 8-field record |
|---|---|---|---|---|
| fp16 | 7.18 | 0.000 | 0.944 | 0.660 |
| q8 per-channel | 7.32 | 0.011 | 0.927 | 0.590 |
| q6 per-channel | 8.90 | 0.236 | 0.861 | 0.305 |
| q4 per-channel | 10.9 | 0.489 | 0.799 | 0.135 |
| q3 per-channel | 26.5 | 1.296 | 0.714 | 0.060 |
| q4 **per-tensor** | 9.6e3 | 8.883 | 0.158 | 0.000 |

- **Harness validation:** the full-precision scores match their planted expectations. Classification is 0.944 against 0.950, and the 8-field record is 0.660 against 0.95^8 = 0.663. Both are within 3 binomial standard errors.
- **Different fates per task:** at q4, classification keeps 85% of its full-precision score but the structured record keeps only 20%. An aggregate hides this.
- **Per-tensor vs per-channel:** with planted outlier rows, per-tensor q4 falls to near chance (0.158), while per-channel q4 scores 0.799.
- **Pure-Python quantiser:** measured RMS error / (step/sqrt 12) = 0.998-1.000 for 3-8 bits.

**Simulated backend (per-task frontier logic):**
- **Aggregate pick:** the rule "cheapest config with aggregate quality >= 0.80" picks q4-chan. At that config perplexity is +18.9%, but json_schema_valid accuracy is 0.023 (97.7% lost) and multistep arithmetic is 0.462.
- **Where the cliff sits:** one step (q5 -> q4) explains more than 90% of json_schema_valid's total drop. The aggregate needs 4 of 5 steps to reach 90%. The per-task pick at a 0.95 bar is q5, one bit wider than the aggregate pick.
- **Pareto frontiers:** 4 distinct frontiers across 5 objectives.
- **Eval-set size:** telling 0.90 from 0.85 needs n >= 686 items per config.

### Limitations

- **Simulated task cliffs:** the cliff in the simulated backend comes from planted per-item margins. The measurement code detects it; it does not prove LLMs behave this way. The real tiny-MLP results show per-task divergence, but not a sharp 5-bit vs 4-bit cliff.
- **Non-monotone curve:** on the tiny MLP, perplexity and KL are *not* monotone in bit width. q5 (KL 0.552) is worse than q4 (KL 0.489) at this seed. A single seed and a tiny model are noisy, so fewer bits is not guaranteed to be worse.
- **Latency:** tiny-MLP latency is wall-clock on dequantised floats. It does not reflect integer-kernel speed and is reported only. Simulated latency and memory use a stated cost model, not target hardware.
- **Not implemented:** no lm-eval-harness, DuckDB or Quarto. The frontier is text output, not a chart.

## Interview questions this project answers

- **Why is perplexity a poor proxy for task performance?**
- **How would you choose a quantisation level for a fixed hardware budget?**
- **How do you benchmark latency reliably on a laptop?**

## What this deliberately is *not*

- Not a quantisation method. It measures the ones that exist, properly.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library**, so there is nothing to install. Run both commands from the repo root.

## Getting started

```bash
git clone https://github.com/SlateGitOrg/quantfrontier.git
cd quantfrontier
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Nothing to install - the runnable reference has no dependencies.

<details>
<summary>Target workflow for the full build (not yet implemented)</summary>

These commands describe the production stack this project grows into.
None of them work in this repository today.

```bash
# git clone <your-fork-url> quantfrontier
# cd quantfrontier
# pip install -e .
# make sweep MODEL=qwen2.5-7b   # q8, q6, q5, q4, q3
# python -m src.eval battery
# quarto render report/frontier.qmd
# pytest
```

</details>

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
