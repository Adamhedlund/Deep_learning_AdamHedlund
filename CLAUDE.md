# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Coursework repo for a Swedish deep learning course (AI25). It is a collection of Jupyter notebooks, not an application: there is no build system, test suite, or linter. Prose (schedule, comments) is largely in Swedish.

- `AI25-Deep_learning/Lectures/` — course lecture notebooks (L1 PyTorch basics → L12 multimodal; topics: NLP/transformers/interpretability, CNNs, transfer learning, RL/PPO, agentic AI, GANs, diffusion). Files are named `L<n><letter>-<Topic>.ipynb`.
- `AI25-Deep_learning/Data/` — images and datasets used by the lectures (CTA ridership CSV, SAE feature JSONs for Gemma/Llama interpretability, diagrams).
- `AI25-Deep_learning/Resources/` — reference links; `schedule.md` — course schedule; `test.ipynb` — scratch notebook.
- `code-along_mnist.ipynb` (repo root) — the student's own code-along work (Keras MNIST).
- `dl-course/` — the student's self-paced PyTorch fundamentals course (plan in `dl-course/plan.md`). When working there, follow the teaching rules in `dl-course/CLAUDE.md`.

## Environment

- Python virtualenv at `.venv/` (gitignored); dependencies in the root `requirements.txt`. Note that it pins **TensorFlow/Keras** (with CUDA 12 NVIDIA wheels), while the lectures also use PyTorch (L1 onwards), so some notebooks may need packages not listed there.
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run notebooks with Jupyter or the VS Code notebook UI against the `.venv` kernel.

## Conventions

- Notebooks reference data via paths relative to `AI25-Deep_learning/` (e.g. `Data/...`), so run/open them with that directory in mind.
- Gitignored and expected to exist only locally: `Lectures/datasets`, `local`, `*.pth`, `Resources/keras`, `*.zip`, `.env` (downloaded datasets, model weights, and secrets such as API keys for the LLM notebooks). Don't commit these.
- Lecture notebooks are course-provided material; put personal work in separate notebooks (like the root code-along) rather than editing lectures unless asked.
