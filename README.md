# medbench-eval

**LLM-as-judge evaluation for medical AI — built on the MedBench-Agent-95 benchmark.**

Evaluate any LLM's clinical capabilities across 13 agentic medical tasks using structured rubrics and automated scoring. Works as a standalone Python tool, a Claude Code skill (`/medbench-eval`), and a CI/CD pipeline component.

---

## What is this?

`medbench-eval` is a framework for evaluating medical AI models using the **LLM-as-judge** methodology, calibrated against the **MedBench-Agent-95** gold-standard benchmark (390 samples across 13 clinical task categories).

**Key design decisions:**

- **Substance over style**: the judge ignores formatting, phrasing, and step ordering — only clinical correctness matters
- **Task-specific rubrics**: each of the 13 tasks has 4-5 Likert criteria designed for that task's clinical requirements
- **Few-shot calibration**: show the judge gold examples before each evaluation to anchor its understanding of score-5
- **RAG-based calibration**: BM25 or embedding retrieval selects the most semantically similar gold examples per question
- **Capability-first UX**: group tasks by what clinical capability you're testing, not by task name
- **DUT tracking**: every evaluation report identifies the model under test alongside the judge model
- **Model-agnostic judge**: supports `claude-haiku-4-5`, `minimax-m2.5`, and `deepseek-chat`

---

## Benchmark Tasks

| Capability | Tasks | What it evaluates |
|------------|-------|-------------------|
| Clinical Reasoning | MedCOT, MedDecomp, MedPathPlan | Differential diagnosis, task decomposition, treatment pathway planning |
| Long-context Understanding | MedLongQA, MedLongConv | Long document Q&A, multi-turn conversation memory |
| Agentic Tool Use | MedCallAPI, MedRetAPI, MedDBOps | API calls, knowledge retrieval, database queries |
| Multi-system Orchestration | MedCollab | Coordinating multiple agents for complex medical goals |
| Self-reflection | MedReflect | Identifying and correcting clinical errors |
| Role Adaptation | MedRoleAdapt | Adapting communication to patient/nurse/physician roles |
| Safety & Defense | MedShield, MedDefend | Refusing harmful requests, resisting adversarial inputs |

Each task has a gold-standard answer used to calibrate what "correct" looks like. The judge scores 1-5 per criterion; scores are normalized to 0-100.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/medqa-auto-eval-skills
cd medqa-auto-eval-skills

# 2. Install
pip install -r requirements.txt

# 3. Set API key (choose one)
export ANTHROPIC_API_KEY=sk-ant-...        # claude-haiku-4-5 (default)
# or configure .env (see Installation)

# 4. Run the interactive wizard
python eval.py
```

---

## Installation

**Requirements:** Python 3.11+, zero mandatory third-party packages for core functionality.

```bash
pip install -r requirements.txt
```

**Judge model options:**

| Model | Env var | Notes |
|-------|---------|-------|
| `claude-haiku-4-5` (default) | `ANTHROPIC_API_KEY` | Fast, cost-effective |
| `minimax-m2.5` | `MINIMAX_API_KEY` + `MINIMAX_BASE_URL` | Via SiliconFlow — OpenAI-compatible |
| `deepseek-chat` | `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL` | Via DeepSeek API — OpenAI-compatible |

You can store all keys in a `.env` file at the project root:

```ini
# .env
MINIMAX_API_KEY=sk-...
MINIMAX_BASE_URL=https://api.siliconflow.cn/v1
MINIMAX_MODEL=Pro/MiniMaxAI/MiniMax-M2.5

DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

The `.env` file is loaded automatically at startup and is excluded from git.

---

## Usage

### Interactive wizard (recommended)

```bash
python eval.py
```

Prompts for DUT name, capability group, sample count, response source, and paths.

### Batch evaluation

```bash
# By capability group
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability reasoning \
  --dut gpt-4o \
  --responses-dir outputs/gpt-4o/ \
  --samples 5 \
  --output results/gpt-4o-reasoning/

# By single task
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --task MedCollab \
  --dut my-model \
  --responses-file outputs/my-model.jsonl \
  --output results/

# Full benchmark
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability full \
  --dut claude-opus-4-6 \
  --responses-dir outputs/opus/ \
  --samples 10 \
  --output results/opus-full/

# Use minimax as judge instead of default claude-haiku-4-5
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --capability safety \
  --dut my-model \
  --responses-dir outputs/ \
  --model minimax-m2.5 \
  --output results/

# With few-shot calibration (recommended for MedCollab, MedDBOps, MedShield)
python eval.py batch \
  --benchmark medbench-agent-95/ \
  --task MedCollab \
  --dut my-model \
  --responses-dir outputs/ \
  --calibrate-n 2 \
  --calibrate-mode bm25 \
  --output results/
```

**Capability group keys:** `reasoning`, `long_context`, `tool_use`, `orchestration`, `self_correction`, `role_adapt`, `safety`, `full`

### Single response evaluation

```bash
python eval.py single \
  --task MedCOT \
  --dut my-model \
  --question "Patient, 45M, fever 39°C, neck stiffness 3 days..." \
  --response "Based on the presentation, I suspect bacterial meningitis..." \
  --gold-answer "Step 1: Key features include fever, meningismus..."
```

Output is JSON with per-criterion scores, minor/major errors, and overall feedback.

### List tasks and capabilities

```bash
python eval.py tasks
```

---

## DUT Response Input Formats

The `--responses-dir` or `--responses-file` flag accepts several formats:

**Directory** (`--responses-dir`): one `.txt` per sample, organized by task and sample ID:
```
outputs/my-model/
  MedCOT/97.txt
  MedCOT/103.txt
  MedCollab/10.txt
```

**Compact JSONL** (`--responses-file`): one record per line:
```jsonl
{"task": "MedCOT", "id": 97, "response": "..."}
{"task": "MedCollab", "id": 10, "response": "..."}
```

**Full benchmark JSONL** (`--responses-file`): same schema as `medbench-agent-95/*.jsonl`:
```jsonl
{"question": "...", "answer": "...", "other": {"id": 97, "source": "MedCOT"}}
```

**Delimited TXT** (`--responses-file`): sections separated by a header line:
```
=== MedCOT | 97 ===
The patient presents with...
=== MedCollab | 10 ===
Agent 1 (Diagnosis) receives...
```

---

## Few-Shot Calibration

The judge can be shown gold answer examples before each evaluation to anchor its score-5 understanding. This significantly improves alignment on tasks where the judge misunderstands the task type.

```bash
# Show 2 gold examples before each sample (random selection)
python eval.py batch ... --calibrate-n 2

# BM25 semantic retrieval — picks the most similar gold example per question
python eval.py batch ... --calibrate-n 2 --calibrate-mode bm25

# Embedding-based retrieval (needs MINIMAX_API_KEY for BAAI/bge-m3)
python eval.py batch ... --calibrate-n 2 --calibrate-mode embedding
```

**When to use:**
- `--calibrate-mode bm25` is recommended for MedCollab, MedDBOps, and MedShield — zero extra dependencies, uses character-level BM25 with CJK-aware tokenization
- `--calibrate-mode random` (default) works well for most other tasks
- `--calibrate-mode embedding` provides the highest quality retrieval but requires a SiliconFlow API key

---

## Judge Alignment Validation

Before running a full evaluation, you can validate that the judge aligns with human expectations:

```bash
# Mode 1 — Synthetic tier test (does the judge distinguish gold from wrong answers?)
python validate.py \
  --benchmark medbench-agent-95/ \
  --all-tasks --samples 5

# Mode 2 — Real model comparison (does the judge reproduce the known human score gap?)
python validate.py \
  --benchmark medbench-agent-95/ \
  --task MedCollab \
  --compare-dir results-gpt-4.1/MedBench_Agent \
  --compare-name gpt-4.1 --compare-expected-score 81 \
  --samples 6 --calibrate-n 2 --calibrate-mode bm25
```

**Metrics reported:**

| Metric | Target | Description |
|--------|--------|-------------|
| `ceiling_score` | ≥ 85 | Avg judge score on gold answers |
| `discrimination_rate` | ≥ 80% | % of samples where gold beats DUT |
| `spearman_rho` | ≥ 0.70 | Rank correlation of expected vs actual order |
| `gap_ratio` | ~1.0× | Actual gap / expected gap |

---

## Scoring

| Component | Range | Description |
|-----------|-------|-------------|
| Criterion score | 1-5 | Per-criterion Likert score (4-5 criteria per task) |
| Total score | 1.0-5.0 | Average of criterion scores |
| Normalized score | 0-100 | `(total - 1) / 4 × 100` |

**Normalized score interpretation:**

| Score | Interpretation |
|-------|----------------|
| 90-100 | Exceptional — matches or exceeds gold standard |
| 75-89 | Strong — minor gaps only |
| 60-74 | Adequate — some important gaps |
| 40-59 | Weak — significant deficiencies |
| 0-39 | Poor — major clinical failures |

### Robustness policy

The judge explicitly separates:
- **Minor errors** (no score impact): different phrasing, formatting, step ordering, synonym terminology, extra correct detail
- **Major errors** (score impact): wrong diagnosis, wrong drug/dose, missing safety-critical steps, incorrect lab interpretation, logical inconsistency

Both lists appear in every evaluation result.

---

## Output Files

After a batch run, `--output results/` contains:

| File | Description |
|------|-------------|
| `summary.json` | Per-task stats, DUT, judge model, error counts |
| `details.jsonl` | One record per sample: scores, criteria, minor/major errors, token usage |
| `report.md` | Human-readable markdown report with tables and per-criterion breakdown |

---

## Repository Structure

```
medqa-auto-eval-skills/
├── README.md
├── SKILL.md                    ← Claude Code /medbench-eval skill
├── eval.py                     ← CLI entry point (wizard + batch + single)
├── validate.py                 ← Judge alignment validation (synthetic tiers & real model compare)
├── requirements.txt
├── judge/
│   ├── capabilities.py         ← 8 capability groups → task mapping
│   ├── rubrics.py              ← 13 task rubrics (57 criteria total)
│   ├── judge.py                ← LLM-as-judge core + robustness policy
│   ├── models.py               ← claude-haiku-4-5 / minimax-m2.5 / deepseek-chat abstraction
│   ├── runner.py               ← batch evaluation loop
│   ├── input_parser.py         ← DUT response parser (directory / JSONL / delimited TXT)
│   ├── rag.py                  ← BM25 + embedding retrieval for calibration anchors
│   └── report.py               ← JSON + markdown report generation
├── references/
│   └── rubrics.md              ← Full rubric table (for Claude Code skill use)
└── medbench-agent-95/          ← benchmark data (390 samples, 13 tasks)
    ├── MedCOT.jsonl + .md
    └── ... (13 tasks × 30 samples each)
```

---

## Task Rubrics

Each task is evaluated on 4-5 criteria. Summary:

| Task | Criteria |
|------|----------|
| MedCOT | step_completeness, clinical_accuracy, differential_diagnosis_quality, evidence_integration, conclusion_clarity |
| MedCallAPI | api_selection, parameter_completeness, format_correctness, error_handling |
| MedCollab | agent_interaction_depth, task_decomposition, workflow_logic, clinical_correctness |
| MedDBOps | query_correctness, clinical_alignment, operation_completeness, data_safety |
| MedDecomp | completeness, logical_ordering, granularity, clinical_feasibility |
| MedDefend | adversarial_detection, refusal_quality, safety_rationale, safe_alternative |
| MedLongConv | memory_accuracy, consistency, clinical_continuity, response_relevance |
| MedLongQA | answer_accuracy, completeness, source_grounding, clarity |
| MedPathPlan | guideline_adherence, individualization, completeness, temporal_logic |
| MedReflect | error_identification, correction_quality, reasoning_depth, improvement_actionability |
| MedRetAPI | query_precision, query_completeness, retrieval_strategy, result_relevance |
| MedRoleAdapt | role_fidelity, communication_style, clinical_accuracy, empathy_professionalism |
| MedShield | risk_recognition, intervention_timeliness, intervention_appropriateness, harm_prevention_effectiveness |

Full criterion definitions (with score-1 and score-5 anchors) are in `judge/rubrics.py` and `references/rubrics.md`.

---

## Benchmark Data

The `medbench-agent-95/` directory contains 390 samples from the [MedBench](https://medbench.opencompass.org.cn/) evaluation suite — 30 samples per task, each with:

- `question`: clinical scenario or task prompt
- `answer`: gold standard response
- `other`: metadata (`id`, `source`)

---

## Claude Code Skill

Add this repo to Claude Code and use `/medbench-eval` directly in your session:

```
/medbench-eval
```

The skill (defined in `SKILL.md`) guides you through the same wizard interactively. Claude acts as the judge inline for single-response evaluation, or delegates to the Python CLI for batch runs.

---

## Contributing

Contributions welcome:
- **New task rubrics** — add to `judge/rubrics.py` and `judge/capabilities.py`
- **New judge models** — add a backend to `judge/models.py`
- **Benchmark extensions** — add JSONL files to `medbench-agent-95/`

---

## License

MIT
