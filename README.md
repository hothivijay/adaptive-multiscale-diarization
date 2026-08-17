# Context-adaptive temporal scale weighting for overlap-aware far-field speaker diarization

Reference implementation of the method in:

> V. Hothi and N. B. Gohil, *Context-adaptive temporal scale weighting for
> overlap-aware far-field speaker diarization.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/synthetic%20pipeline%20test-passing-brightgreen)

---

## What the method does

Multi-scale diarization analyses audio at several window lengths at once, because
short windows track speaker changes precisely while long windows give stable speaker
identities. In the configurations shipped with the standard toolkit and used as
challenge baselines, the weight given to each window length is **uniform, or fixed
for a whole recording** — so a session whose acoustics change halfway through is
served by a compromise setting throughout.

This work makes that combination rule a function of local acoustic context. A small
convolutional scorer reads only the surrounding embedding context and re-estimates
the reliability of all five scales **at every time index**; a softmax across scales
turns those scores into weights, and the resulting convex combination is decoded into
overlap-aware multi-label speaker activity.

No speech separation. No overlap annotation. No prior speaker count.

```
                 ┌──────────────┐
  waveform ─────►│ MarbleNet VAD│──► speech regions V          (frozen, 0.31 M)
                 └──────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┬───────────────┐
     1.50 s          1.25 s          1.00 s          0.75 s          0.50 s   ← 50 % overlap
        │               │               │               │               │
        └──────────► TitaNet-Large embeddings, 192-dim ◄─────────────────┘
                        │                                        (frozen, 23 M)
                        ├──► NME-SC per scale ──► speaker count k, prototypes
                        │                                          (Eqs. 1–2)
                        ▼
              ┌────────────────────────┐
              │ reliability scorer     │  r_s(t) ← 2×Conv1d(C=32,k=3) on 2c+1 context
              │ softmax ACROSS SCALES  │  w_s(t) = softmax_s r_s(t)     (Eqs. 3–5)
              └────────────────────────┘         ▲ weights shared over s
                        │                        │ ~58 k params
                        ▼
                z(t) = Σ_s w_s(t)·E_s(t)         convex combination   (Eq. 6)
                        │
                        ▼
              ┌────────────────────────┐
              │ 3-layer LSTM, drop 0.5 │  h(t) = LSTM(z(t), h(t−1))    (Eq. 7)
              │ N independent sigmoids │  p_n(t) = σ(v_nᵀh(t) + c_n)   (Eq. 8)
              └────────────────────────┘         ▲ v_n from prototypes — see note
                        │                          ~1.56 M params
                        ▼
              threshold 0.5 → smooth → RTTM
```

The quantity that separates this from a fixed-weight system is the **temporal
variance of the weight sequence** (Eq. 10): identically zero under session-constant
weighting, strictly positive under context-conditioned weighting. The implementation
exports it at inference so the distinction can be checked rather than assumed.

---

## Reported results

Full tables in [`paper/REPORTED_RESULTS.md`](paper/REPORTED_RESULTS.md). Zero-collar
DER with overlap scored, on the DISPLACE 2024 evaluation set:

| System | Eval DER (%) |
| --- | --- |
| DISPLACE-2024 official baseline | 29.96 |
| pyannote.audio 3.1 | 34.96 |
| Strongest published challenge submission | 26.70 |
| **Proposed system** | **21.46** |

**28.37 %** relative reduction against the official baseline; **19.63 %** against the
strongest published submission.

The controlled ablation, in which the weighting rule is the only variable:

| Configuration | Eval DER (%) |
| --- | --- |
| + Multi-scale decoder, **static** weights | 25.76 |
| + Multi-scale decoder, **dynamic** weights | **21.46** |

4.30 pp from a module holding about 6 % of the parameters — larger than the combined
3.21 pp of upgrading the VAD and the embedding extractor. That is the paper's central
claim, and it is the part this repository reproduces.

> **Honest framing.** The numbers above are the paper's. This repository writes its
> own tables to `results/tables/RESULTS.md`, and `scripts/06_ablation_and_stats.py`
> prints both columns side by side. Differences belong in your write-up, not
> reconciled away.

---

## Quick start

### Verify the repository without the corpus or NeMo

DISPLACE 2024 is access-controlled and NeMo is a ~3 GB install, so neither is needed
to check that everything works:

```bash
pip install torch numpy scipy scikit-learn PyYAML
python tests/test_pipeline_synthetic.py
```

This fabricates a corpus with known speaker structure and drives the full pipeline —
segmentation, cross-scale alignment, NME-SC, prototypes, targets, **both ablation
arms**, RTTM emission, zero-collar DER, Algorithm 2 — asserting the invariants the
paper states:

```
2. Parameter budget vs Table 9
  [PASS] scorer+decoder ~1.5 M              -- 1,621,283
  [PASS] complete system ~24.81 M           -- 24,931,283
  [PASS] trainable share ~6 %               -- 6.5%
3. Training both ablation arms
  [PASS] Eq.(10) V_s == 0 exactly for static     -- V_s=0.000e+00
  [PASS] Eq.(10) V_s > 0 for trained dynamic     -- V_s=4.354e-03
4. Eq.(5) invariants on a trained dynamic model
  [PASS] weights sum to 1 across scales
  [PASS] weights non-negative
  [PASS] weights vary over time
5b. Algorithm 1 line 15 round-trip (deterministic)
  [PASS] emits a segment per active speaker      -- ['A', 'B']
  [PASS] isolated single frame smoothed out
  [PASS] A boundaries within one frame of truth  -- [2.50, 7.75] vs [2.50, 7.75]
  [PASS] overlap representable: A and B concurrent
  [PASS] RTTM write/read round-trips exactly
7. Eq.(12) dual relative reporting
  [PASS] Eq.(12) vs baseline == 28.37 %          -- 28.37%
  [PASS] Eq.(12) vs best published == 19.63 %    -- 19.63%

ALL CHECKS PASSED
```

The DER implementation is separately validated against seven hand-computed cases
(perfect match with overlap, total miss, label permutation, false alarm, speaker
collapse, missed overlap, partial detection).

### Full experiment

Open a notebook and run it top to bottom:

- [`notebooks/lightning_pipeline.ipynb`](notebooks/lightning_pipeline.ipynb) — **recommended**
- [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb)

Or from a shell:

```bash
pip install -r requirements.txt
pip install "nemo_toolkit[asr]==2.0.0"        # front end only, stage 02

python scripts/01_prepare_data.py             # manifest + Table 4 checks
python scripts/02_extract_embeddings.py --dry-run --limit 3    # cost estimate
python scripts/02_extract_embeddings.py       # 1–2 GPU-h, ONCE, resumable
python scripts/03_cluster_and_targets.py      # NME-SC, prototypes, targets

python scripts/04_train.py --mode static      # ablation arm 1
python scripts/04_train.py --mode dynamic     # ablation arm 2

python scripts/05_infer_and_score.py --run static  --partition eval
python scripts/05_infer_and_score.py --run dynamic --partition eval

python scripts/07_estimate_snr.py
python scripts/06_ablation_and_stats.py --static static --dynamic dynamic
python scripts/08_parameter_budget.py
python scripts/09_figures.py
```

---

## Where to run it, and what it costs

**The corpus is the blocker, not compute.** DISPLACE 2024 is released on request by
its organisers under a data usage policy — no script here downloads it, and none
should. See [`docs/DATASET.md`](docs/DATASET.md).

**Compute is far cheaper than the paper's 20 hours suggests**, and the reason is
architectural. The front end is *frozen* (Section 3.8), so a frozen network on fixed
audio gives the same answer every time. Recomputing it per epoch, per ablation arm
and per seed is pure waste. So the pipeline splits:

| stage | cost | how often |
| --- | --- | --- |
| upload corpus (~4.4 GB) | one-off | **once, ever** |
| extract 5-scale embeddings | 1–2 GPU-hours | **once, ever** — resumable |
| cluster + build targets | minutes, CPU | once |
| train one arm | **minutes** | per arm, per seed |
| score + tables + figures | minutes, CPU | as needed |

The arithmetic behind the storage plan:

```
segments/second, summed over the five shifts
    1/0.750 + 1/0.625 + 1/0.500 + 1/0.375 + 1/0.250 = 11.6
total segments   ≈ 11.6 × 38 h × 3600 s            ≈ 1.59 M
cache (float16)  ≈ 1.59e6 × 192 × 2 bytes          ≈ 610 MB
with audio and the rest                            ≈ 5–6 GB
```

float16 is deliberate: the embeddings feed a cosine affinity and a convex
combination, neither sensitive at the fifth decimal, and halving 1.2 GB to 610 MB is
the difference between comfortable and tight on a 15 GB Drive.

### Platform comparison

| | persistent storage | free GPU | verdict |
| --- | --- | --- | --- |
| **Lightning.ai Studio** | `/teamspace` survives stop/start | monthly allowance | **recommended** — corpus and cache persist natively, no Drive mount, local-disk read speed, and you can drop to a free CPU machine between GPU stages |
| **Google Colab** | none; needs Drive mount (15 GB free) | T4, session-limited | works — extraction is resumable, so disconnects cost only the session in flight |
| **Kaggle** | `/kaggle/input` read-only | generous weekly hours | good GPU overflow — attach the corpus as a Dataset once; derived data needs re-publishing as versions |

All three cost **$0**. The differentiator is persistence, which is why Lightning is
first. `src/amsd/platform.py` detects the platform and routes every artefact to the
right storage class, so the same code runs unmodified on all of them:

```python
from amsd.platform import mount_persistent_storage
paths = mount_persistent_storage()
# platform      : lightning
# persistent root: /teamspace/studios/this_studio/amsd-displace2024
```

Override anywhere with `AMSD_ROOT=/any/path`.

**Resumability is the design point.** Each session's cache is marked complete only
after every array is written, so a truncated write is never mistaken for a finished
one. Re-run the same command after a disconnect and finished sessions are skipped.

---

## One thing you should read before trusting any number

Eq. (8) as printed is **under-specified**, and the literal reading does not work.

`p_n(t) = σ(v_nᵀh(t) + c_n)` reads as though `v_n` were a globally learned vector.
But speaker identity is session-scoped: speaker 0 of session 1 and speaker 0 of
session 2 are different people in different regions of the embedding space, so a
global `v_0` is trained toward contradictory targets and can only converge to the
session-averaged prior — silence.

This was observed, not theorised. Under the literal reading the smoke test gives:

```
epoch 6/6  train 0.5643  val 0.5837     ← pinned at the majority-class value
corpus DER = 100.00%  (miss 100.0 / fa 0.0 / conf 0.0)
```

PIT is the usual escape, but Section 2.4 explicitly positions this work *against*
PIT-based EEND. The alternative is the one the paper itself states in Section 3.2:

> "The five streams **and their cluster prototypes** are then passed to the dynamic
> weighting module."

So `v_n` is taken to be the session's NME-SC prototype for unit `n`. After the
change, training loss falls from 0.61 to 0.029 and the model learns. No oracle
information enters: prototypes come from embeddings and clustering, the count from
Eq. (2), and nothing reads the reference RTTM.

Set `decoder.prototype_conditioned: false` to reproduce the literal reading and its
failure.

**Fifteen further decisions and manuscript inconsistencies are catalogued in
[`docs/PAPER_GAPS.md`](docs/PAPER_GAPS.md)** — including the base time grid for
Eq. (6), the justification for `hidden_size=256` (verifiable with
`scripts/08_parameter_budget.py --sweep-hidden`), and the fact that **the statistical
table Algorithm 2 specifies is absent from the paper**: Section 4.2 says "Table 9
gives the outcome" but Table 9 is the computational profile. This repository
generates it.

---

## Repository layout

```
├── configs/
│   ├── base.yaml                    every Table 2 value, plus marked PAPER-SILENT choices
│   ├── ablation_static.yaml         arm 1 — w_s(t) = 1/S, V_s ≡ 0
│   └── ablation_dynamic.yaml        arm 2 — w_s(t) from the scorer, V_s > 0
├── src/amsd/
│   ├── platform.py                  Colab / Lightning / Kaggle / local path routing
│   ├── config.py                    dataclasses; PAPER-SILENT notes at each definition
│   ├── data/
│   │   ├── rttm.py                  RTTM I/O, overlap statistics
│   │   ├── segmentation.py          5-scale windows + cross-scale alignment
│   │   ├── targets.py               multi-label Y, speaker↔unit binding, RTTM emission
│   │   └── dataset.py               chunked reader over the cached embeddings
│   ├── frontend/                    ── all frozen, all cached ──
│   │   ├── vad.py                   MarbleNet, 0.3 s overlap tolerance, hysteresis
│   │   ├── embeddings.py            TitaNet-Large, resumable float16 memmap cache
│   │   ├── clustering.py            NME-SC, Eqs. (1)–(2), from scratch in NumPy
│   │   └── prototypes.py            cluster prototypes for Eq. (8)
│   ├── model/
│   │   ├── scorer.py                Eqs. (3)–(5), (10) ← the contribution
│   │   ├── decoder.py               Eqs. (7)–(8) + the prototype note
│   │   └── system.py                the static/dynamic ablation switch
│   ├── metrics/
│   │   ├── der.py                   Eqs. (11)–(12), zero collar, Hungarian mapping
│   │   ├── stats.py                 Algorithm 2 + seed variance
│   │   └── conditions.py            Tables 7–8 stratification
│   ├── train.py                     Eq. (9), Adam 1e-3, cosine annealing
│   └── infer.py                     Algorithm 1 lines 6–11, 15 + V_s export
├── scripts/01…09                    the pipeline, one stage per script
├── notebooks/                       Colab and Lightning, runnable top to bottom
├── tests/test_pipeline_synthetic.py end-to-end, no corpus, no NeMo
├── tools/push_to_github.py          secrets-based push; refuses corpus data
├── docs/DATASET.md                  how to obtain and lay out the corpus
├── docs/PAPER_GAPS.md               every implementation decision, justified
└── paper/REPORTED_RESULTS.md        the paper's tables, verbatim, for comparison
```

---

## Notes on fidelity

**The two ablation arms differ in exactly one configuration field.** Same cached
embeddings, same NME-SC partition, same prototypes, same targets, same decoder, same
Eq. (9), same optimiser and schedule, same 0.5 threshold, same smoother — the same
code reading the same files. That is what licenses attributing the difference to the
weighting rule.

**Eq. (10) is computed in float64.** In float32 the static arm gives V_s ≈ 3.4e-16 —
one ULP of 0.2, squared — rather than zero. float64 accumulation gives exactly 0.0,
which keeps the diagnostic a clean binary test instead of a threshold comparison.

**Zero-collar DER is implemented directly**, not shelled out to md-eval, so the
per-session list Algorithm 2 needs is available without parsing scorer logs.
`T_total` counts overlapped regions once *per concurrent speaker*, as Eq. (11)
requires — dividing by wall-clock speech instead is the commonest way to report a
DER that looks better than it is.

**Training and inference speech regions are kept strictly separate**, as Figure 8
requires: oracle regions from the reference during training, MarbleNet at inference.
Using oracle regions at test time would remove every missed-speech error and produce
a number not comparable with the challenge baselines.

**`scripts/09_figures.py` produces `fig13`, which the paper does not have.** Figure 7
in the manuscript is explicitly a schematic — "drawn to explain the mechanism and are
not measured outputs". `fig13` plots the *real* `w_s(t)` exported at inference,
averaged over reference speaker-change points. It may not reproduce the schematic's
shape, which is precisely why it is worth looking at.

---

## Known limitations

Inherited from the paper (Section 4.7) and not fixed by this implementation:

1. **Single-run evidence.** The paper trained each configuration once, so its
   ablation carries no variance over training runs. Because extraction is cached,
   extra seeds cost only decoder training — `scripts/04_train.py --seed` and
   `scripts/06_ablation_and_stats.py --seeds` close this gap cheaply. Doing so is the
   single highest-value addition to this work.
2. **One corpus.** Nothing here establishes generalisation beyond DISPLACE 2024.
3. **Hard conditions stay hard.** Error remains above 30 % at 5–10 dB SNR and above
   30 % overlap. The weighting mechanism redistributes emphasis among embeddings; it
   cannot create speaker evidence none of them contains.
4. **Offline only.** Session-level spectral clustering and non-causal smoothing both
   prevent streaming. The reported real-time factor describes batch throughput, not
   latency.

Specific to this implementation:

5. **Never run against DISPLACE 2024.** Every verification here is on synthetic data
   or hand-computed cases, because the corpus is access-controlled. The code is
   correct on everything that can be checked without it; the first real run may still
   surface corpus-specific issues.
6. **Table 6 rows 1–3 are not reproducible here** — they need the official DISPLACE
   baseline recipe.

---

## Citation

```bibtex
@article{hothi_adaptive_multiscale_diarization,
  title   = {Context-adaptive temporal scale weighting for overlap-aware
             far-field speaker diarization},
  author  = {Hothi, Vijay and Gohil, Narendrasinh B.},
  note    = {Gujarat Technological University; Government Engineering College, Rajkot}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT for the software — see [`LICENSE`](LICENSE). This does **not** cover the DISPLACE
2024 corpus, which carries its own data usage policy, nor the pretrained MarbleNet
and TitaNet-Large checkpoints, which NVIDIA distributes under its own terms.
