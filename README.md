# VLA Lab

**Felipe Barbosa · Stanford · CS project submission**

A **living notebook** on vision-language-action models: what I trained on real hardware, what broke, what I learned, and an interactive site that tries to make the forward pass legible — especially for people who are not already deep in ML.

**Run the site locally:**

```bash
git clone https://github.com/fbarbosa-stanford/vla-lab.git
cd vla-lab
python3 -m http.server 8124
# open http://localhost:8124
```

**Weights, datasets, and eval rollouts:** [huggingface.co/fbarbosa1](https://huggingface.co/fbarbosa1)

---

## What this project is

Three things together:

1. **Robotics work** — datasets I collected, models I finetuned, and logged examples of policies working and failing on an SO-101 arm (and open-loop driving clips from a π₀.5 AV finetune).
2. **A public log of learning** — written while I was still confused about VLAs myself, so my younger brother (incoming freshman) and friends outside AI could follow along later.
3. **This website** — Explorer, methods notes, model zoo, and driving write-up that **show** behavior through clips and diagrams instead of dumping benchmark tables into the UI.

---

## Project Submission Rubric (15 Points)

### Problem & Insight (3 Points)

**Problem.** I had a hard time understanding VLAs and other large robotics models from papers and checkpoints alone. It was unclear what each part of the stack actually *did* — vision tokens, language conditioning, flow-matching action heads, chunking — and why a policy could look fine in notebooks but fail on the arm.

**Motivation.**

- I wanted a **living blog** of what I was learning as I went, not a polished post-hoc summary.
- My brother is an incoming freshman; I wanted him to have a **readable log** of my robotics experiments if he ever picks up the thread.
- I also wanted to **prove to myself** that I understood the model — by explaining it, training it, breaking it, and publishing both.

**Insight.** The same VLA recipe spans embodiments (arm vs. car): instruction + image + state → action chunk. If I could explain that pipeline once, I could reuse the language for manipulation and driving. The site is the explanation; Hugging Face holds the artifacts.

---

### Execution & Technical Work (5 Points)

#### Datasets (LeRobot format on Hugging Face)

| Dataset | What it is |
|---------|------------|
| [so101_embodied_tasks_v1](https://huggingface.co/datasets/fbarbosa1/so101_embodied_tasks_v1) | ~100 teleop episodes on an SO-101 — fork pick-up and lego-into-cup tasks, top-down + wrist cameras |
| [so101_lego_into_glass](https://huggingface.co/datasets/fbarbosa1/so101_lego_into_glass) | Teleop demos for a narrower lego-into-glass task (ACT baseline) |
| [eval_act_so101_lego_into_glass](https://huggingface.co/datasets/fbarbosa1/eval_act_so101_lego_into_glass) | **Logged policy rollout** on hardware — the arm missing the glass (used as a failure example on the site) |

Collected and exported with [LeRobot](https://github.com/huggingface/lerobot) conventions (parquet + video, v3 schema).

#### Models trained & published

| Model | Base | Method | Task |
|-------|------|--------|------|
| [smolvla_so101_embodied_v1](https://huggingface.co/fbarbosa1/smolvla_so101_embodied_v1) | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | BC finetune, frozen vision | SO-101 embodied tasks |
| [pi05_so101_embodied_v1](https://huggingface.co/fbarbosa1/pi05_so101_embodied_v1) | [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | BC finetune, flow-matching action expert | SO-101 embodied tasks |
| [act_so101_lego_into_glass_020000_v6](https://huggingface.co/fbarbosa1/act_so101_lego_into_glass_020000_v6) | — | Action Chunking Transformer | Lego into glass |
| [pi05-nvidia-av-generalize-012000](https://huggingface.co/fbarbosa1/pi05-nvidia-av-generalize-012000) | π₀.5 stack | BC on NVIDIA PhysicalAI AV clips | Ego-frame driving (open-loop eval) |

Training stack: **LeRobot** (`lerobot-train`), Modal GPU jobs for longer runs, local SO-101 eval loops with remote inference servers. I finetuned **SmolVLA** (0.5B VLA) and **π₀.5** (4B flow-matching VLA) on my own data, plus an **ACT** baseline for comparison on a simpler task.

#### Methods I actually used

- **Behavior cloning** on teleop datasets with frozen vision encoders
- **LoRA / action-expert-only** training where full finetunes were too heavy
- **Flow-matching action heads** (π₀.5) — denoising an action chunk conditioned on tokens
- **Action chunking** (ACT) — predict H steps, execute, re-infer
- **Open-loop diagnostics** — replay training frames through a checkpoint to separate “bad policy” from “bad deployment”
- **Closed-loop logging** — record failed (and partial) real-robot episodes back to HF as eval datasets

The [Methods](methods.html) page explains these in plain language (including GRPO/DPO as general post-training ideas, not as part of this driving submission).

#### Examples on the site (not benchmark dashboards)

I **deliberately avoided** pasting ADE tables and leaderboard numbers into the website. Instead:

- **When something worked** — partial grasps, sensible open-loop driving overlays, Explorer demos that match the architecture I trained.
- **When something failed** — embedded HF eval video (ACT missing the glass), qualitative notes on SmolVLA undershoot and π₀.5 task confusion, driving clips where the predicted path drifts.

That matches how I validated claims: look at **episodes**, not a single scalar on the homepage.

#### The website (communication layer)

| Page | Role |
|------|------|
| [Explorer](explorer.html) | Architecture-faithful **teaching** sim — patch tokens, action queries, flow τ, chunk heatmap, SO-101 motion (does not load 4B weights in-browser). The **step-by-step animation** and **ASCII diagrams** on the site were **AI-generated**, then reviewed by me for architectural accuracy. |
| [Methods](methods.html) | Field guide to finetuning decisions I had to make |
| [Models](models.html) | Zoo + **“What breaks”** with HF clips |
| [Driving](driving.html) | π₀.5 AV finetune + held-out open-loop clip gallery |
| [Home](index.html) | Map of the pipeline |

**Iteration.** I showed early drafts to friends who are not into AI and rewrote copy, added the staged Explorer reveal, and replaced jargon-heavy sections with the failure gallery — so the site reads like an explanation, not a lab report.

---

### Evaluation & Evidence (3 Points)

| Question | How I answered it |
|----------|-------------------|
| Did I train real policies? | Public HF checkpoints + training configs |
| Do they transfer to hardware? | SO-101 rollouts; eval dataset with video |
| Do they always work? | **No** — failure examples published on purpose |
| Do I understand the architecture? | Explorer + methods text tied to what I trained |
| Does driving work closed-loop? | **Not claimed** — open-loop clip examples only |

**Failure analysis (on the site and Hub).**

- ACT lego: logged eval — policy never completes insertion
- SmolVLA: partial fork/lego success; undershoot and placement sensitivity
- π₀.5 manipulation: language-conditioned; task bleed when scene is ambiguous
- Driving: qualitative open-loop overlays on held-out AV footage

**Limitations.**

- Explorer internals are illustrative, not attention maps from my checkpoints
- Explorer **animation** and site **ASCII art** (homepage pipeline, methods diagrams, page headers) were **generated with AI** — they teach the architecture, not logged telemetry from my runs
- No on-vehicle closed-loop deployment in this project
- GRPO/DPO driving experiments belong to a **separate** project and are not part of this repo

---

### Communication & Presentation (2 Points)

**Audience.** Course staff, robotics-curious friends, and my brother — people who may never run `lerobot-train` but should still understand *what a VLA is* and *what I did*.

**This README vs. the rest of the project.** The site and Hugging Face artifacts are the **living log** — messy, iterative, updated as I learned. **This README is the polished submission** for the course: it summarizes the robotics work, evidence, and limitations in one place for graders.

---

### Process, Integrity & Disclosure (2 Points)

**AI tools.** Training and debugging these models required AI-assisted tooling — reading LeRobot configs, tracing tensor shapes, Modal/infra scripts, and understanding which submodule (vision vs. action expert vs. processor) was responsible when runs failed. AI assistants also helped with site copy and layout, the **Explorer step animation**, and **ASCII diagrams** across the site (homepage pipeline, methods page, page headers). **This README was drafted and polished with AI help** from my notes on what I actually trained, logged, and broke; I reviewed every section so the claims match the artifacts on Hugging Face and the site. **All training choices, eval logging, and claims about what is real vs. simulated were made and reviewed by me.** This repository is authored by Felipe Barbosa only.

**Credits & citations.**

| Resource | Role |
|----------|------|
| [LeRobot](https://github.com/huggingface/lerobot) | Dataset schema, training CLI, policy implementations, eval conventions |
| [SmolVLA](https://huggingface.co/papers/2506.01844) / [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | Compact VLA baseline I finetuned |
| [Physical Intelligence π₀ / π₀.5](https://www.physicalintelligence.company/blog/pi05) / [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | Flow-matching VLA stack for manipulation and driving finetunes |
| [ALOHA](https://tonyzhaozh.github.io/aloha/) (Tony Zhao et al.) | Teleop / bimanual manipulation lineage that popularized the data-collection workflow this builds on |
| [Three.js](https://threejs.org/) | Explorer 3D scene (vendored) |
| NVIDIA PhysicalAI AV data | Driving finetuning source |
| SO-101 follower arm | Manipulation hardware |

**Artifacts.** This GitHub repo, Hugging Face models/datasets, and commit history on both.

---

## Repo structure

```
vla-lab/
├── index.html          # Homepage
├── explorer.html       # Interactive VLA Explorer
├── methods.html        # Finetuning guide
├── models.html         # Model zoo + failure examples
├── driving.html        # Self-driving note + clip gallery
├── css/  js/  vendor/  # Site assets
└── assets/driving/     # Open-loop driving preview clips
```

---

## License

Site content © Felipe Barbosa. Model weights on Hugging Face use the license stated on each model card (Apache-2.0 where applicable).
