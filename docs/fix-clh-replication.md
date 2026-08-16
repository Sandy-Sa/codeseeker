> **SUPERSEDED — see [`clh-replication-status.md`](clh-replication-status.md).**
> The parser fixes below all landed and worked (analyse recall 0.0055 → 0.505).
>
> **The target numbers in this document are CORRECT.** CLH-base on the full 70K
> label space is **0.32 micro / 0.14 macro**, exactly as stated below. A
> 2026-08-13 revision of `clh-replication-status.md` "corrected" this to
> 0.42 / 0.27 — that was wrong, and it has since been reverted; see §1 of that
> document. The replication hit **0.326 / 0.152** on the test split, i.e. it met
> the target as originally written here.
>
> The claim below that the raw `completions` endpoint was a secondary issue was
> right in kind but understated: it turned out to be the primary remaining gap,
> and fixing it is what closed the replication.
>
> Kept for the diagnosis history only.

# Fixing the CLH / MDACE replication (near‑zero F1 → paper numbers)

**Audience:** an autonomous coding agent (Claude Code) working in this repo.
**Goal:** raise the MDACE‑ICD10CM pipeline from its current broken state
(Overall micro‑F1 ≈ **0.009**) to the paper's range (CLH‑base, full 70K label
space: **micro‑F1 ≈ 0.32 / macro‑F1 ≈ 0.14**, from *Code Like Humans*, EMNLP
Findings 2025, arXiv:2509.05378). ✅ *Achieved: 0.326 / 0.152 — see
[`clh-replication-status.md`](clh-replication-status.md).*

> Work top‑down. **Step 1 (analyse parser) is the single highest‑leverage fix.**
> Verify each step before moving on. Do **not** burn a GPU job until the offline
> parser unit test (Step 1c) passes.

---

## 1. Background: what the pipeline does and what broke

The pipeline is a 4‑agent ICD‑10 coder run as a DVC chain (see `dvc.yaml`):

```
analyse  → locate → verify → assign
(retrieval) (filter) (filter) (final filter)
```

- **analyse** asks the LLM to extract clinical terms from the note, embeds them,
  and retrieves candidate ICD‑10 codes from a Qdrant index of the ICD‑10 trie.
  This stage sets the **recall ceiling** — downstream stages only *filter*.
- **locate / verify / assign** ask the LLM to select/keep candidate IDs.

### The failure (evidence)

From the last run `.pbs/2026-06-28-15-04-16ef5b07/`:

| stage | **Overall** f1_micro | **Overall** recall | note |
|---|---|---|---|
| analyse | 0.0042 | **0.0055** | recall ceiling set here |
| locate | 0.0056 | 0.0055 | inherited |
| verify | 0.0059 | 0.0055 | inherited |
| assign | 0.0091 | 0.0049 | final |

`run.e` shows the proximate cause directly:

```
[0/10] Could not find <answer> tags in response: ...are:\n- NSTEMI\n- SAH\n- CAD\n- HTN...
   Retrying with increased temperature and tokens: (0.648, 15000)
...
[Analyse Agent] Results: Average extracted snippets: 1.14, Average extracted codes: 8.61
```

The model (**DeepSeek‑R1‑Distill‑Llama‑70B**) produces *correct* clinical terms,
but emits them as a markdown/newline list after `</think>` **without** the
required `<answer>…</answer>` tags. The strict parser rejects ~every response,
retries up to 10× at rising temperature (which makes formatting worse), and the
stage ends with **1.14 search snippets/doc** (should be ~10–15). With almost no
queries, retrieval recovers only ~16 of ~2,870 gold codes → 0.55% recall →
everything downstream is capped.

### Two metric definitions — use the right one

`evaluate_and_dump_metrics` (`experiments/utils.py:362`) prints **[Overall]** and
**[Contextual]**:

- **[Overall]** = predictions vs *all* document gold codes. **This is the
  paper‑comparable metric. Optimize this.**
- **[Contextual]** = predictions vs `subset_targets`, where `subset_targets` =
  gold codes that were *also retrieved* (`experiments/analyse_agent.py:231`).
  At the analyse stage the prediction set *is* the retrieved set, so Contextual
  recall is trivially ≈1.0. **It is self‑referential; ignore it for replication.**

---

## 2. Confirmed NOT the cause — do not chase these

These were investigated and ruled out. Don't spend time here.

- **ICD‑9 codes in the parquet.** `mdace_inpatient_annotations.parquet` mixes
  systems in one `code` column, disambiguated by `code_type`
  (`icd10cm: 3329, icd9cm: 3292, icd10pcs: 263, icd9pcs: 253`). The loader
  filters to `icd10cm` **before** building `targets`
  (`src/dataloader/mdace/generator.py:185‑190`), and `format_dataset`
  (`experiments/utils.py:52‑60`) drops anything not in the ICD‑10 trie. Verified
  gold is full dotted ICD‑10‑CM (`E11.9`, `J45.909`, `N18.9`, `I10`; 93% dotted,
  the 7% 3‑char are legit billable codes). **Gold data is correct.**
- **Code format / dotted‑vs‑undotted mismatch.** The trie, predictions, and gold
  are all dotted ICD‑10‑CM. No mismatch.
- **Wrong model.** The paper's *CLH‑base* IS DeepSeek‑R1‑Distill‑Llama‑70B. The
  model choice is correct; the *plumbing around it* (parsing + endpoint) is wrong.

---

## 3. Root causes (in priority order)

1. **Strict `<answer>` parsers reject the reasoning model's real output.**
   All four agents hard‑require `<answer>` tags with an exact shape and raise
   `StructuredError` otherwise:
   - analyse: `src/agents/analyse_agent.py:34` (`ANSWER_PATTERN`, line 14)
   - locate: `src/agents/locate_agent.py:16` (`ANSWER_PATTERN`, line 10)
   - verify: `src/agents/verify_agent.py:71` (`ANSWER_PATTERN`, line 10)
   - assign: `src/agents/assign_agent.py:16` (`ANSWER_PATTERN`, line 9)
2. **Raw `completions` endpoint skips DeepSeek's chat template.** `params.yaml:44`
   sets `model.endpoint: completions`; `base.py:108‑122` only sends structured
   `messages` (which vLLM renders with the model's chat template) on
   `chat/completions`. On `completions` the model gets a flat string with no
   `<｜Assistant｜><think>` priming, degrading instruction‑following.
   ⚠️ Caveat: `completions` was chosen deliberately so `max_tokens` is honored
   (`_config.py:136‑140`). See Step 2 for how to switch safely.
3. **Retry loop amplifies the failure** (`base.py:182` →
   `sync_structured_batch_call(..., max_retries=10)`): each parse failure retries
   at higher temperature. Resolves once parsing is robust.

---

## 4. Fix plan

### Step 0 — Orient and set up cheap iteration

1. Read the four agent files end‑to‑end and confirm the line numbers above
   (they may have drifted): `src/agents/{analyse,locate,verify,assign}_agent.py`
   and `src/agents/base.py`.
2. Note how each `agent_type` in `params.yaml` maps to a class via the
   `create_*_agent` factories (e.g. analyse `base` → `BaseAnalyseAgent`;
   locate `split` → `LocateSplitAgent`; verify/assign `reasoning`).
3. The DVC stages depend on the agent source files, so editing them will
   correctly invalidate and re‑run only the affected stages.

### Step 1 — Make the **analyse** parser robust (PRIMARY FIX)

This recovers the recall ceiling. Two complementary changes:

**1a. Add a no‑`<answer>` fallback** in `BaseAnalyseAgent.parser`
(`src/agents/analyse_agent.py:34`). When `ANSWER_PATTERN` misses, strip the
`<think>…</think>` block and parse the trailing list. Strip markdown bullets
(`-`, `*`, `1.`), bold (`**`), and drop full‑sentence lead‑ins
(e.g. lines starting with "The"/"These", or ending in "."). Proposed shape:

```python
def parser(self, content: str) -> dict[str, typ.Any]:
    m = re.search(ANSWER_PATTERN, content, re.DOTALL)
    if m:
        raw = self._normalise_raw(m.group(1))
    else:
        # Reasoning models often emit the list after </think> with no <answer>
        # tags. Recover it instead of discarding a good response.
        body = re.sub(THINKING_PATTERN, "", content, flags=re.DOTALL)
        raw = self._normalise_raw(self._recover_list(body))
    if not raw:
        raise StructuredError(f"Empty answer in response: {content[-250:]}")
    terms = self._tokenise(raw)
    cleaned = {self._clean_term(t) for t in terms if self._is_term(t)}
    cleaned = {t for t in cleaned if t}
    if not cleaned:
        raise StructuredError(f"Could not parse response: {raw[-250:]}")
    return {"reasoning": content, "output": sorted(cleaned)}
```

Helpers to add (adapt as needed):

```python
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

@staticmethod
def _clean_term(t: str) -> str:
    t = BaseAnalyseAgent._BULLET.sub("", t)
    t = t.replace("**", "").strip().strip('"').strip("'")
    return t.rstrip(".").rstrip(",").strip()

@staticmethod
def _is_term(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:                 # drop prose
        return False
    low = s.lower()
    if low.startswith(("the ", "these ", "here ", "based on", "note:")):
        return False
    return True

@staticmethod
def _recover_list(body: str) -> str:
    # Keep only the lines that look like list items.
    lines = [ln for ln in body.splitlines()
             if BaseAnalyseAgent._is_term(ln)]
    return "\n".join(lines)
```

> The captured responses in `run.e` render spaces as `Ġ` and newlines as `Ċ`
> (byte‑level BPE artifacts of the logger). Decode them (`Ġ`→space, `Ċ`→`\n`)
> when building test fixtures.

**1b. Tighten the prompt** so the model is more likely to emit `<answer>` in the
first place. In `src/agents/templates/analyse_agent/strict_v3.yml.j2`, make the
final instruction explicit and example‑driven, e.g.:

```
After your reasoning, output ONLY the terms, one per line, wrapped exactly like:
<answer>
term one
term two
</answer>
```

Keep the fallback from 1a regardless — reasoning models are not 100% reliable.

**1c. Prove it offline (no GPU)** before any DVC run. Create
`tests/agents/test_analyse_parser.py` with **real** failing responses copied from
`.pbs/2026-06-28-15-04-16ef5b07/run.e` (decode `Ġ`/`Ċ`). Example:

```python
import pytest
from agents.analyse_agent import create_analyse_agent  # or import BaseAnalyseAgent

CASES = [
    # (raw model content, expected terms ⊆ parsed output)
    ("...are:\n\n- NSTEMI\n- SAH\n- CAD\n- HTN\n- Hyperlipidemia\n- Obesity\n\n"
     "These conditions are explicitly documented and relevant.",
     {"NSTEMI", "SAH", "CAD", "HTN", "Hyperlipidemia", "Obesity"}),
    ("reasoning...\n</think>\n\nThe current conditions extracted are:\n\n"
     "abdominal pain  \ntachycardia  \npancreatic pseudocyst  \nSIRS  \n"
     "pancreatitis  \nobesity",
     {"abdominal pain", "tachycardia", "pancreatic pseudocyst", "SIRS",
      "pancreatitis", "obesity"}),
]

@pytest.mark.parametrize("content,expected", CASES)
def test_recovers_terms_without_answer_tags(content, expected):
    agent = BaseAnalyseAgent.__new__(BaseAnalyseAgent)  # parser is pure
    out = set(agent.parser(content)["output"])
    assert expected <= out                 # all real terms recovered
    assert not any("documented" in t for t in out)  # no prose leaked
```

Run: `PYTHONPATH=src:experiments python -m pytest tests/agents/test_analyse_parser.py -q`.
Iterate on the helpers until every captured case passes and no prose leaks.

### Step 2 — Switch to the chat endpoint (recommended, do carefully)

This gives DeepSeek its chat template (proper `<｜Assistant｜><think>` priming),
which markedly improves format compliance.

1. In `params.yaml`, set `model.endpoint: chat/completions` (the exact literal —
   see the `Literal[...]` in `_config.py:30`).
2. **Resolve the `max_tokens` caveat** (`_config.py:136‑140`): verify that, on
   `chat/completions`, the throughster vLLM interface still forwards a token cap
   that vLLM honors (it may require `max_completion_tokens` instead of, or in
   addition to, `max_tokens`). Inspect throughster's request builder and, if
   needed, send both. **Confirm generations are not truncated before `<answer>`**
   (watch for responses that end mid‑`<think>`).
3. Re‑check `base.py:108‑122`: on `chat/completions` it sends `prompt.messages`.
   Confirm the `strict_v3` template still renders sensible `system`/`user`
   messages; the trailing `<think>` in the user turn is now redundant (the chat
   template adds it) — harmless, but you may remove it.

If the `max_tokens` issue can't be cleanly resolved, **keep `completions`** and
rely on Steps 1 + 3 — the parser fixes alone recover most of the gap.

### Step 3 — Apply the same robustness to locate / verify / assign

These select candidate **integer IDs** inside `<answer>` (not free text), so the
fix differs from analyse. `run.e` shows the model emitting IDs without tags
(`**Answer:** 5, 6, 7`), codes instead of IDs, or the literal placeholder
`<answer>...</answer>`.

Preferred: **use the structured / guided‑decoding variants the repo already has.**
`StructuredLocateAgent` (`src/agents/locate_agent.py:97‑104`) sets
`sampling_params["guided_regex"]` to force a parseable integer list — parsing
then cannot fail. Plan:

1. Inspect `create_locate_agent` / `create_verify_agent` / `create_assign_agent`
   for a `structured`/guided `agent_type`. If one exists, switch the stage in
   `params.yaml` to it (e.g. `agents.locate.agent_type: structured`).
2. If a guided variant is missing for verify/assign, add one mirroring
   `StructuredLocateAgent`: set a `guided_regex` (e.g. comma‑separated ints)
   and a trivial `parser` that splits the constrained output.
3. Guided decoding requires the request to pass `guided_regex` (or
   `guided_json`) through to vLLM. The existing `StructuredLocateAgent` proves
   this path works with this throughster build; reuse it.

Fallback (if not using guided decoding): relax each `parser` to also accept a
trailing comma/space‑separated integer list after `</think>` when `<answer>`
tags are absent, and treat an empty/placeholder `<answer>` as `output: []`
(assign already does this at `assign_agent.py:28‑29`). Add captured‑output unit
tests as in Step 1c for each agent.

### Step 4 — Re‑run and compare to the paper

1. Re‑run the full chain on the test split:
   `dvc repro assign@mdace-icd10cm` (editing agent source invalidates the chain;
   add `-f`/`-s` if you need to force a specific stage).
2. Read the printed **[Overall]** metrics per stage from the job logs.

---

## 5. Acceptance criteria

Check in order; do not proceed if an earlier gate fails.

- [ ] **Offline parser tests pass** (Step 1c, and Step 3 if added) with the real
      captured outputs — terms/IDs recovered, no prose leakage.
- [ ] **analyse stage**: `Average extracted snippets` ≈ **8–15/doc** (was 1.14)
      and **[Overall] recall ≳ 0.6** (was 0.0055). *This is the make‑or‑break gate
      — locate/verify/assign can only lower recall.*
- [ ] **assign (final) [Overall]**: micro‑F1 in the **same order of magnitude as
      the paper** — target ≈ 0.32 micro / 0.14 macro for the full 70K space
      (`retrieval.all_codes: true`). Anything ≥ ~0.2 confirms the pipeline works;
      ~0.009 means it's still broken. Exact parity isn't expected (prompt/version
      drift), but it must not be ~zero.
- [ ] The 10× retry storm in `run.e` is gone (few/no `Could not find <answer>`
      warnings).

If analyse recall is healthy but final F1 is still low, the bug has moved to a
downstream filter (Step 3) — debug locate/verify/assign next, not analyse.

---

## 6. File map (quick reference)

| Concern | Location |
|---|---|
| analyse parser (PRIMARY) | `src/agents/analyse_agent.py:34` (`ANSWER_PATTERN` :14, `THINKING_PATTERN` :15, `_tokenise` :65) |
| analyse prompt | `src/agents/templates/analyse_agent/strict_v3.yml.j2` |
| locate parser / guided variant | `src/agents/locate_agent.py:16` / `:97` |
| verify parser | `src/agents/verify_agent.py:71` |
| assign parser | `src/agents/assign_agent.py:16` (empty‑answer ok at `:28`) |
| request build / endpoint switch | `src/agents/base.py:102‑122` |
| endpoint config + max_tokens caveat | `params.yaml:39‑52`, `experiments/stages/_config.py:30`, `:136‑140` |
| metric (Overall vs Contextual) | `experiments/utils.py:362`; `subset_targets` def `experiments/analyse_agent.py:231` |
| gold/target construction (verified OK) | `src/dataloader/mdace/generator.py:185‑190`, `src/dataloader/adapt/adapters/mdace.py:55` |
| retry loop | `src/agents/base.py:180‑184` |

## 7. Pitfalls

- **Don't trust [Contextual].** Optimize and report **[Overall]**.
- **Don't truncate generations.** If switching to chat, verify the token cap is
  honored or reasoning gets cut before `<answer>` (re‑introducing the bug).
- **Prose leakage.** When recovering a tag‑less list, exclude explanatory
  sentences ("These conditions are…") — assert this in tests.
- **Iterate offline first.** Each full GPU run is expensive; the parser unit
  tests catch ~all of this without vLLM.
- **One lever at a time.** Land Step 1, confirm analyse recall jumps, *then*
  do Steps 2–3, so you can attribute changes.
