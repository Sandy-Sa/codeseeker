# CLH / MDACE replication — COMPLETE

**Status: replicated.** Run `.pbs/2026-08-15-19-11-c2938349` (commit `03778e4`)
reproduces the paper's CLH-base full-label-space result. Locked in `dvc.lock`.

**Paper:** *Code Like Humans: A Multi-Agent Solution for Medical Coding*,
Findings of EMNLP 2025, pp. 22612–22627.
[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1231/) ·
[arXiv:2509.05378](https://arxiv.org/abs/2509.05378).

---

## 1. The correct target

> **Correction (2026-08-16).** An earlier revision of this document asserted the
> full-70K target was **0.42 / 0.27** and "corrected" `fix-clh-replication.md`'s
> 0.32 / 0.14. That was backwards. **`fix-clh-replication.md` was right.**
> No 0.42 / 0.27 row exists anywhere in Table 1. Sanity check that should have
> caught it: the 70K space is strictly harder than the constrained 1K space, so
> its score must fall *below* CLH-base's 1K row (0.38 / 0.24) — it cannot exceed
> CLH-large's 0.43 / 0.28. **Always transcribe from the published PDF**
> (`aclanthology.org/2025.findings-emnlp.1231.pdf`), not the arXiv HTML render
> and not from memory.

Paper **Table 1** (p. 22617), transcribed verbatim:

| Model | #P | #C | F1 Micro | F1 Macro | EMR |
|---|---|---|---|---|---|
| *constrained label space (prior work)* | | | | | |
| PLM-ICD | 340M | 1K | 0.48 | 0.25 | 0.02 |
| PLM-ICD | 340M | 6K | 0.46 | 0.21 | 0.02 |
| Llama3-70B† | 70B | 1K | 0.28 | 0.18 | 0.01 |
| CLH-small | 8B | 1K | 0.27 | 0.18 | 0.02 |
| CLH-base | 70B | 1K | 0.38 | 0.24 | 0.02 |
| CLH-large | 235B | 1K | 0.43 | 0.28 | 0.02 |
| CLH-o3-mini | – | 1K | 0.37 | 0.24 | 0.02 |
| CLH-o4-mini | – | 1K | 0.41 | 0.27 | 0.02 |
| *full label space (ours, realistic clinical setting)* | | | | | |
| **CLH-base** | **70B** | **70K** | **0.32** | **0.14** | **0.02** |

† = DeepSeek-R1 distilled Llama3-70B.

We run `retrieval.all_codes: true`, so **our target is the last row:
0.32 micro / 0.14 macro / 0.02 EMR.** CLH-base is our model
(DeepSeek-R1-Distill-Llama-70B).

Paper **Table 2** is the per-stage breakdown of **CLH-large (235B) at the
constrained 1K space** — its final row (0.43 / 0.28) is exactly CLH-large's
Table 1 row. It is *not* like-for-like with our per-stage numbers on either axis
(model size *and* label space), so do not read a shortfall from it.

| stage (paper's name) | micro | macro | recall | precision |
|---|---|---|---|---|
| 1. evidence extractor (= our `analyse`) | 0.12 | 0.09 | 0.62 | 0.06 |
| 2. index navigator (= our `locate`) | 0.36 | 0.25 | 0.53 | 0.27 |
| 3. tabular validator (= our `verify`) | 0.40 | 0.27 | 0.47 | 0.34 |
| 4. code reconciler (= our `assign`) | 0.43 | 0.28 | 0.46 | 0.36 |

Paper **Table 5** (Appendix D.3) is directly comparable to our analyse stage:
S-PubMedBert recall@k = 0.50 / **0.54** / 0.60 / 0.64 / 0.68, so
**recall@10 = 0.54**.

---

## 2. Result

Run `.pbs/2026-08-15-19-11-c2938349`, commit `03778e4`, full 70K label space,
DeepSeek-R1-Distill-Llama-70B on 2×H200 via vLLM. Exit 0, 3 h 15 m walltime.

**[Overall] metrics** (`subset_targets`-based *[Contextual]* figures are
self-referential — see §5 — and must never be quoted):

| stage | all 550 notes | | | **test split (n=115)** | | |
|---|---|---|---|---|---|---|
| | micro | macro | recall | **micro** | **macro** | recall |
| analyse | 0.070 | 0.019 | 0.546 | 0.066 | 0.025 | 0.530 |
| locate | 0.261 | 0.091 | 0.457 | 0.255 | 0.109 | 0.448 |
| verify | 0.314 | 0.115 | 0.410 | 0.313 | 0.144 | 0.409 |
| **assign** | **0.325** | **0.122** | 0.390 | **0.326** | **0.152** | 0.388 |

**Head-to-head on the paper's protocol (MDACE test split):**

| | micro-F1 | macro-F1 | EMR |
|---|---|---|---|
| paper, CLH-base 70K | 0.32 | 0.14 | 0.02 |
| **ours, test split** | **0.326** | **0.152** | **0.035** |
| ours, all 550 notes | 0.325 | 0.122 | 0.009 |

Micro-F1 lands on target whichever eval set you use. Macro-F1 and EMR come in
slightly *above* the paper on the test split. The analyse stage's 0.546 recall
matches Table 5's 0.54.

**Caveat on macro-F1.** It is the split-sensitive metric here: 0.152 on the test
split (above target) but 0.122 across all 550 (below). Micro-F1 sits at ~0.325
either way, so lead with micro. Macro is computed only over classes with
TP+FP+FN > 0, so it is dominated by how many *distinct* wrong codes get
predicted, and that varies with sample size.

**What moved the numbers.** Versus the previous run
(`.pbs/2026-06-30-08-17-531549a2`, 0.281 / 0.110, recall 0.303), the endpoint
change (§4.1) took assign to 0.325 / 0.122 and recall to 0.390. Parse retries
across all four stages dropped from **267 to 3**.

---

## 3. Fidelity to the released code

Diffed against a fresh clone of `github.com/MotzWanted/codeseeker`.

**Byte-identical where it matters scientifically:**

- the entire `src/agents/templates/` tree — all four prompts in use
- `experiments/{analyse,locate,verify}_agent.py` — the pipeline step modules
- `TrieClassificationMonitor` and `evaluate_and_dump_metrics` — char-for-char,
  so metrics are computed by upstream's code
- `build_eval_trie` mirrors `tanner_benchmark.py`'s eval-trie construction
- `skip_indices` `[27, 179, 260, 327, 379, 394]` matches `benchmark.py:115`

**Infrastructure-only deviations** (no scientific effect): offline trie load via
`from_dir` instead of the unconditional cms.gov scrape; embedded Qdrant
(`local_path`) instead of a server; consolidated split feather instead of three
CSVs; lazy import of the private `ml-datasets` package; response-cache hardening
(0-byte entries, operation timeout); picklable exceptions across
`datasets.map(num_proc>1)` worker boundaries; explicit `Features` schema in
assign. The DVC decomposition in `experiments/stages/_llm_stage.py` mirrors
`tanner_benchmark.py` call-for-call.

**Deliberate scientific deviations** — the three a reviewer will ask about:

1. **`endpoint: chat/completions` vs upstream's `completions`** (§4.1). The
   largest one, and it is carrying the result.
2. **Rewritten parsers** (`src/agents/parsing.py`). Beyond byte-BPE decoding and
   tag-less recovery, this makes a *semantic* change: an empty answer is a valid
   "nothing applies" (`[]`) rather than a `StructuredError` that triggers
   escalating-temperature retries. That changes filter behaviour, not just
   robustness — the agents can now express a negative.
3. **`icd.year: 2022`** vs `tanner_benchmark.py`'s `2024`, and
   **`sampling.max_tokens: 50000`** vs upstream's `10_000`. Both measured or
   argued harmless (§4.3).

---

## 4. Changes made

### 4.1 `completions` → `chat/completions` (the lever)

`prompt_poet`'s `.string` — what the `completions` endpoint sends — is a bare
concatenation of the system and user text. Verified by rendering the real
template against the real tokenizer:

```
OLD (/v1/completions), what the model actually received:
  'You are a highly skilled medical coding assistant...'
  ... '...within <answer>...</answer>.\n<think>'

NEW (/v1/chat/completions), after vLLM applies the model's own chat template:
  '<｜begin▁of▁sentence｜>You are a highly skilled medical coding assistant...'
  ... '...within <answer>...</answer>.<｜Assistant｜><think>\n'
```

No BOS, no `<｜User｜>`/`<｜Assistant｜>` role tokens. DeepSeek-R1-Distill is
conditioned on exactly those, and its model card requires the chat template.
This also explains the shape of the failure: `analyse` is a robust "list the
terms" task and survived, while `locate`/`verify`/`assign` are long-context,
careful-instruction-following selection tasks — precisely what degrades when a
chat model loses its template.

The paper (Appendix C) states the prompt files are "compatible with both
completion and chat completion endpoints", so this is not a deviation from the
paper — it is the correct path for a chat-template model. It *is*, however, a
deviation from the released code, and it should be stated as such.

- `params.yaml`: `model.endpoint: chat/completions`.
- `src/agents/base.py`: on the chat path, strip the template's hand-rolled
  trailing `<think>` from the last user turn (the chat template appends its
  own). Templates stay byte-identical to upstream, so the endpoint remains a
  pure params flip and the `completions` path is unchanged.
- Tests: `tests/agents/test_prompt_endpoint.py`.

### 4.2 Report the paper's protocol alongside upstream's

`experiments/stages/_metrics.py` scores each stage restricted to the MDACE
**test** split and writes `{stage}_metrics_by_split.json`, using the same
`TrieClassificationMonitor` so the figures come from identical code. It
deliberately does not re-call `evaluate_and_dump_metrics` (that would overwrite
`responses.json` with a subset). Unlike upstream's dump it stores **both**
`overall` and `contextual` under each split, so the Overall numbers survive
(§5). Tests: `tests/stages/test_split_metrics.py`.

### 4.3 Deliberately NOT changed

- **`icd.year` stays 2022.** Measured: the 2022 trie already retains **903/904
  unique MDACE gold codes (99.9%)** and 3328/3329 annotations, so there is no
  recall headroom. Switching to 2024 also currently *fails* — CMS renamed the
  guideline PDFs, so `ICDFileMap.from_directory` cannot resolve `cm_guidelines` /
  `pcs_guidelines`.
- **`sampling.max_tokens` stays 50,000** (upstream 10,000). Strictly more
  permissive; truncation can only hurt a reasoning model.
- **`skip_indices` stays applied** — 6 of 556 rows, matching upstream.
- **locate's empty-answer semantics stay strict.** locate raises on `[]` where
  verify/assign accept it. Held back on purpose so this run attributes cleanly
  to the endpoint change. With retries now down to 3 in total, there is nothing
  left for it to recover — leave it.

---

## 5. Traps

Three ways to read these results wrong. All three have bitten this project.

**Quote `[Overall]`, never `[Contextual]`.** `[Contextual]` scores against
`subset_targets` = gold ∩ retrieved, so its recall is ~1.0 by construction and
its micro-F1 (0.42 for assign) *coincidentally resembles the wrong 0.42 target
from §1*. That coincidence is almost certainly how the bad target got recorded
in the first place.

**`{stage}_metrics.json` on disk contains `[Contextual]`, not `[Overall]`.**
Upstream `evaluate_and_dump_metrics` (`experiments/utils.py:463`) writes both
dicts to the *same* filename in a loop, so the contextual write wins:

```python
for metrics in [overall_metrics, contextual_metrics]:
    with open(dump_path / f"{file_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f)
```

The all-550 **Overall** figures exist only in the stage's stdout (`run.o`).
`{stage}_metrics_by_split.json` (ours) is unaffected — it stores both. Left
as-is to keep upstream's function byte-identical; read the log or the by-split
file instead.

**Directory deps include `__pycache__`.** `src/retrieval`, `src/trie` and
`src/dataloader` are tracked as whole directories in `dvc.yaml`, and DVC hashes
`__pycache__/*.pyc` along with the source. A local tree that has imported these
modules therefore hashes differently from a fresh clone, so **run-cache lookups
miss and `dvc repro` re-executes stages that are genuinely up to date**. The run
hashed `src/retrieval` at 13 files; a clean checkout has 9. Fix is to add
`__pycache__/` to `.dvcignore` — **not done here**, because changing it
re-hashes every dep and would invalidate the whole DAG, forcing a 3 h GPU
re-run. Do it at the start of the next campaign, not the end of this one.

---

## 6. Provenance of `dvc.lock`

`run.sh` clones the repo into `$PBS_JOBFS/work` and runs there, so the
`dvc.lock` the job wrote died with the node. It was reconstructed from DVC's own
**run-cache** (`/scratch/gp50/zs5704/projects/codeseeker/dvc/runs`), which
persists in the shared cache and stores the exact lock stanza DVC computed for
each executed stage. Not hand-written.

Verification performed before committing:

- `stage_data` and `prep` were re-reproduced on the login node; `prep` re-ran
  (see the `__pycache__` trap in §5) and produced a **byte-identical** output
  hash `af83791afbba0629501cb831a5507ec7.dir`, confirming determinism.
- Every code dep in the run-cache stanzas hashes identically in the working
  tree at `03778e4`.
- The dep→out chain was verified link by link: `prep` out → `analyse` dep →
  `analyse` out → `locate` dep → … → `assign` out.
- Every object referenced by the lock exists in the cache, recursively.
- The `assign` metrics the lock points at are the reported ones
  (micro 0.32582, macro 0.15211 on the test split).

Note `metadata.txt` records `git_commit=595972a` — that is the *submit-time*
value. `run.sh` re-resolves `HEAD` at job start, and `run.e` confirms the job
checked out **`03778e4`**. Trust the log, not the metadata file.

---

## 7. Next

MDACE was the replication checkpoint and it is done. The real target is
**MIMIC-III**, evaluated against `~/repos/prompt-icd/`'s
`processed_mimic_iii_full.parquet`. That needs a full-MIMIC-III loader and an
ICD-9 trie (or an ICD-9→10 mapping decision); codeseeker's `mimic-iii-50` loader
covers only the 50-code split.

Before that campaign: add `__pycache__/` to `.dvcignore` (§5), and commit
`dvc.lock` from the compute node rather than reconstructing it — e.g. have
`run.sh` copy `$WORKDIR/dvc.lock` back to `$PBS_O_WORKDIR` on success.
