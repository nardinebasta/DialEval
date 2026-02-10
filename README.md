# DIALEVAL: Instruction-Following Evaluation for LLMs and Dialogue Systems

DIALEVAL is a multi-agent evaluation framework that assesses how well LLM-generated responses follow instructions. It operates through a two-stage **Analyzer → Evaluator** pipeline, decomposing complex prompts into atomic evaluation criteria and then scoring each response against them using an LLM-as-a-judge approach.

The framework supports three evaluation modes:

1. **Single-turn evaluation** — Evaluate a standalone prompt/response pair
2. **Input-dependent evaluation** — Evaluate responses to prompts that include user-provided input text (e.g., summarisation, transformation tasks)
3. **Dialogue-level evaluation** — Evaluate multi-turn conversations where each response is assessed in the context of the preceding message

---

## Table of Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
  - [Mode 1: Single-Turn Evaluation](#mode-1-single-turn-evaluation)
  - [Mode 2: Input-Dependent Evaluation](#mode-2-input-dependent-evaluation)
  - [Mode 3: Dialogue-Level Evaluation](#mode-3-dialogue-level-evaluation)
- [File Descriptions](#file-descriptions)
- [Execution Order](#execution-order)
- [Output Format](#output-format)
- [Instruction Types](#instruction-types)
- [Conversation File Format](#conversation-file-format)
- [Benchmarks](#benchmarks)
- [Citation](#citation)

---

## Architecture

DIALEVAL uses a two-stage pipeline powered by Claude (Anthropic):

```
                          ┌──────────────────────┐
                          │   Analyzer Agent      │
                          │                       │
     Prompt ─────────────►│  Decompose prompt     │──────► Atomic Instructions
     (+ optional input)   │  into atomic,         │        (typed, with dependencies)
                          │  evaluatable criteria  │
                          └──────────────────────┘
                                                              │
                                                              ▼
                          ┌──────────────────────┐    ┌──────────────────┐
                          │   Evaluator Agent     │    │                  │
                          │                       │    │  LLM Response    │
     Atomic Instructions ─►  Score each criterion ◄────  to evaluate     │
                          │  against the response  │    │                  │
                          │  (binary + evidence)   │    └──────────────────┘
                          └──────────────────────┘
                                      │
                                      ▼
                              Evaluation Report
                          (per-instruction scores,
                           per-type scores, overall)
```

**Stage 1 — Instruction Decomposition (Analyzer):**
Breaks a complex prompt into atomic instructions, each classified by type (`content`, `format`, `style`, `logical`, `numerical`) with dependency tracking and optional verifiability flags.

**Stage 2 — Response Evaluation (Evaluator):**
Assesses each atomic instruction against the LLM response, producing binary satisfaction judgments (`true`/`false`) with textual evidence, then aggregates into per-type and overall scores.

---

## Repository Structure

```
dialeval/
├── InstructionAnalyzer.py                  # Core: Atomic instruction decomposition
├── InstructionEvaluator.py                 # Core: Response evaluation against instructions
├── InstructionAnalyzerInputHandeling.py    # Extended: Input-dependent prompt decomposition
├── InstructionEvaluatorInputHandeling.py   # Extended: Input-dependent response evaluation
├── DIalEval_Main.py                        # Dialogue: Multi-turn conversation evaluator
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
```

---

## Installation

### Prerequisites

- **Python 3.9+**
- **Anthropic API key** — obtain from [console.anthropic.com](https://console.anthropic.com)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/dialeval.git
cd dialeval

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

All scripts read the API key from the `ANTHROPIC_API_KEY` environment variable:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows CMD
set ANTHROPIC_API_KEY=your-api-key-here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model (Analyzer) | `claude-sonnet-4-5` | Used for instruction decomposition |
| Model (Evaluator) | `claude-sonnet-4-5` | Used for response evaluation |
| Max Tokens | `4096` | Maximum tokens per API call |
| Temperature | `0` | Deterministic output for reproducibility |

---

## How to Run

### Mode 1: Single-Turn Evaluation

**Files involved:** `InstructionAnalyzer.py` → `InstructionEvaluator.py`

Evaluate a standalone prompt/response pair:

```python
import os
from InstructionAnalyzer import InstructionAnalyzer
from InstructionEvaluator import InstructionEvaluator

api_key = os.environ.get("ANTHROPIC_API_KEY")

# Step 1: Decompose prompt into atomic instructions
analyzer = InstructionAnalyzer(api_key=api_key)
instructions = analyzer.process_prompt(
    "Write a formal letter requesting a refund. Include the order number and reason."
)

# Step 2: Evaluate a response against those instructions
evaluator = InstructionEvaluator(analyzer, api_key=api_key)
evaluation = evaluator.evaluate_response(
    prompt="Write a formal letter requesting a refund. Include the order number and reason.",
    response="Dear Customer Service, I am writing regarding order #12345..."
)

# Step 3: Generate detailed report with per-type scores
report = evaluator.generate_detailed_report(evaluation)
print(json.dumps(report, indent=2))
```

Or run the built-in example:

```bash
python InstructionEvaluator.py
```

### Mode 2: Input-Dependent Evaluation

**Files involved:** `InstructionAnalyzerInputHandeling.py` → `InstructionEvaluatorInputHandeling.py`

For prompts that depend on user-provided input text (e.g., "Summarise the following article"):

```python
import os
from InstructionAnalyzerInputHandeling import InstructionAnalyzer
from InstructionEvaluatorInputHandeling import InstructionEvaluator

api_key = os.environ.get("ANTHROPIC_API_KEY")
analyzer = InstructionAnalyzer(api_key=api_key)
evaluator = InstructionEvaluator(analyzer, api_key=api_key)

prompt = "Choose an appealing title for your post."
user_input = "The typical avocado is over 300 calories from the oil in it..."
response = "What Will Happen If You Eat an Avocado Every Day?"

# Analyze with input context
instructions = analyzer.process_prompt(prompt, user_input)

# Evaluate considering both prompt and input
evaluation = evaluator.evaluate_response(prompt, response, user_input, instructions)
report = evaluator.generate_detailed_report(evaluation)
```

Or run the built-in example:

```bash
python InstructionEvaluatorInputHandeling.py
```

### Mode 3: Dialogue-Level Evaluation

**Files involved:** `DIalEval_Main.py` (self-contained)

Evaluate multi-turn conversations where each victim/bot response is assessed in the context of the preceding scammer/adversary message:

```python
import os
from DIalEval_Main import DialevalTester

api_key = os.environ.get("ANTHROPIC_API_KEY")

victim_prompt = """You are an AI scam-baiter with the goal of wasting
as much time as possible for the scammer..."""

tester = DialevalTester(api_key, victim_prompt)

# Evaluate all conversations organised by LLM in subdirectories
tester.run_evaluation(
    conversations_dir="datasets/conversations",
    output_base_dir="results"
)
```

Or edit the paths in `main()` and run directly:

```bash
python DIalEval_Main.py
```

---

## File Descriptions

### Core Pipeline (Single-Turn)

| File | Role | Stage |
|------|------|-------|
| **`InstructionAnalyzer.py`** | Decomposes a prompt into atomic instructions. Each instruction is classified by type (`content`, `format`, `style`, `logical`, `numerical`), assigned an ID, and linked via dependencies. Validates output structure. | **Stage 1: Analysis** |
| **`InstructionEvaluator.py`** | Takes atomic instructions + an LLM response and evaluates each instruction as satisfied (`true`) or not (`false`) with supporting evidence. Computes per-type and overall scores. Imports from `InstructionAnalyzer`. | **Stage 2: Evaluation** |

### Extended Pipeline (Input-Dependent)

| File | Role | Stage |
|------|------|-------|
| **`InstructionAnalyzerInputHandeling.py`** | Extended version of `InstructionAnalyzer` that handles input-dependent prompts. Includes multiple prompt templates: standard (no input), input-dependent (with user input), and verifiability-aware variants. Contains additional prompt engineering for human-aligned evaluation. | **Stage 1: Analysis** |
| **`InstructionEvaluatorInputHandeling.py`** | Extended version of `InstructionEvaluator` that evaluates responses considering user-provided input. Adds verifiable/non-verifiable instruction scoring and character-level formatting verification. Imports from `InstructionAnalyzerInputHandeling`. | **Stage 2: Evaluation** |

### Dialogue Evaluation

| File | Role | Stage |
|------|------|-------|
| **`DIalEval_Main.py`** | Self-contained dialogue evaluation system. Includes its own Analyzer and Evaluator prompts tailored for dialogue context. Parses conversation files, evaluates each victim response against the preceding scammer message, generates per-conversation Excel workbooks and per-LLM aggregate statistics. | **Stages 1 & 2 combined** |

---

## Execution Order

### For single-turn evaluation:
```
1. InstructionAnalyzer.py    →  Produces atomic instructions (JSON)
2. InstructionEvaluator.py   →  Evaluates response against instructions → scores
```

### For input-dependent evaluation:
```
1. InstructionAnalyzerInputHandeling.py   →  Produces atomic instructions with input context
2. InstructionEvaluatorInputHandeling.py  →  Evaluates response with input awareness → scores
```

### For dialogue evaluation:
```
1. DIalEval_Main.py  →  Analyzes prompt once, then evaluates each turn → Excel reports
```

The analysis step (Stage 1) runs **once per prompt**. The evaluation step (Stage 2) runs **once per response** to evaluate.

---

## Output Format

### Evaluation Report

```json
{
    "overall_score": 0.73,
    "type_scores": {
        "content": 0.80,
        "format": 0.50,
        "style": 1.00,
        "logical": 0.67,
        "numerical": 0.00
    },
    "instruction_evaluations": [
        {
            "instruction_id": 1,
            "instruction": "Include key dates related to World War II",
            "type": "content",
            "satisfied": true,
            "evidence": "Response includes: 1939, September 1 1939, 1940, 1941..."
        },
        {
            "instruction_id": 7,
            "instruction": "Use proper headings in the response",
            "type": "format",
            "satisfied": false,
            "evidence": "The response is a single paragraph with no headings."
        }
    ]
}
```

### Dialogue Evaluation Outputs (per LLM)

| Output File | Contents |
|-------------|----------|
| `instructions.json` | Atomic instructions extracted from the system prompt |
| `<conversation>.xlsx` | Per-turn satisfaction matrix + evidence + conversation text |
| `<llm>_overall_stats.json` | Aggregate scores (overall, per-type, per-instruction) |
| `<llm>_detailed_stats.xlsx` | Multi-sheet Excel: Summary, All Instructions, per-type breakdowns |

---

## Instruction Types

| Type | Description | Examples |
|------|-------------|---------|
| `content` | Information and content requirements | "Include key dates", "Mention major battles" |
| `format` | Structural and formatting requirements | "Use headings", "Present in chronological order" |
| `style` | Writing style and tone requirements | "Academic tone", "Formal language", "Below 30 words" |
| `logical` | Reasoning and logic requirements | "Support claims with evidence", "Maintain logical flow" |
| `numerical` | Mathematical and quantitative requirements | "Calculate total casualties", "Include statistics" |

---

## Conversation File Format

For dialogue evaluation (`DIalEval_Main.py`), conversation files should follow this format:

```
LLM: <model_name>
Scenario: <description>
Scammer: Hello, I'm calling from your bank regarding suspicious activity...
Victim: Oh hello, who's calling please?
Scammer: This is the fraud department, we need to verify your account...
Victim: Oh dear, what seems to be the problem?
```

Organise files in subdirectories by LLM:

```
datasets/conversations/
├── gpt4/
│   ├── conversation_001.txt
│   ├── conversation_002.txt
│   └── ...
├── claude/
│   └── ...
└── llama/
    └── ...
```

---

## Benchmarks

DIALEVAL has been validated against two established instruction-following benchmarks:

- **[InfoBench](https://arxiv.org/abs/2401.03601)** — Expert-annotated instruction-following evaluation with decomposed questions and multi-annotator agreement
- **[IFEval](https://arxiv.org/abs/2311.07911)** — Instruction-following evaluation with verifiable constraints (Google)

---

## Notes

- **Model compatibility:** JSON parsing handles both raw JSON and markdown-fenced output (`\`\`\`json ... \`\`\`\``), ensuring compatibility with Claude 3.5 Sonnet through Claude Sonnet 4.5.
- **API costs:** Each evaluation requires 2 API calls (1 analysis + 1 evaluation). Dialogue evaluation requires 1 analysis + N evaluations (one per turn).
- **Reproducibility:** Temperature is set to `0` for deterministic outputs.
- **Rate limits:** For large-scale evaluation, consider adding retry logic with exponential backoff.

---

## Citation

If you use DIALEVAL in your research, please cite:

```bibtex
@misc{dialeval2025,
  title={DIALEVAL: Instruction-Following Evaluation for LLMs and Dialogue Systems},
  author={Basta, Nardine},
  year={2025}
}
```

---

## License

[Add your chosen licence here]
