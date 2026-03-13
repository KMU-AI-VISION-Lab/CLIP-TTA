# AGENTS.md

This repository contains experimental extensions for geometry analysis of CLIP embeddings built on top of the CLIP-TTA codebase.

All automated agents (Codex, GPT, etc.) must follow these rules when modifying the repository.

---

# 1. General Development Principles

- Treat this repository as a research codebase.
- Prioritize **minimal, safe changes**.
- Avoid refactoring unrelated code.
- Preserve backward compatibility with existing experiments.
- Prefer **adding new modules or scripts** rather than modifying core training code.

---

# 2. Scope of Changes

Allowed:
- Adding new analysis metrics
- Adding visualization tools
- Adding evaluation utilities
- Adding new CLI flags
- Adding new scripts

Avoid:
- Refactoring training pipeline
- Changing solver behavior
- Changing dataset loading logic unless explicitly requested

---

# 3. Code Style

- Keep functions small and readable
- Prefer explicit variable names
- Add comments explaining research logic
- Avoid unnecessary abstractions
- Prefer numpy / pytorch utilities already used in the repo

---

# 4. Experiment Safety

When adding new functionality:

- Do not break existing commands
- Make new features **opt-in via CLI flags**
- Add clear logging
- Validate tensor shapes
- Handle small-sample edge cases gracefully

---

# 5. Output Requirements

All new evaluation utilities must:

- Print clear stdout summaries
- Save results to JSON
- Include dataset name, backbone, and selected classes
- Avoid overwriting existing outputs

---

# 6. Visualization Rules

Visualization is **qualitative only**.

UMAP or other dimensionality reduction should:

- Be optional
- Never replace original-space metrics
- Save figures instead of only displaying them

---

# 7. Geometry Analysis Philosophy

Quantitative conclusions should rely on **original embedding space metrics**, including:

- centroid similarity
- retrieval accuracy
- pairwise distance correlation
- neighborhood topology metrics
- same-dataset split-half upper bounds

Visualization is used only for interpretability.

---

# 8. Commit Philosophy

Changes should follow **small logical commits**:

Examples:
- fix: correct pairwise distance correlation
- feat: add class geometry retrieval accuracy
- feat: add same-dataset upper-bound evaluation
- feat: add topology metrics and visualization

---

# 9. Execution Reporting

When an agent modifies code it must report:

- files changed
- commands to run the new functionality
- expected output format