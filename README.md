# VLA Lab

**Felipe Barbosa · Stanford University · CS Project Submission**

Vision-language-action (VLA) models map camera images and language instructions to robot motion. This project trains and evaluates VLAs on real hardware, publishes the artifacts on Hugging Face, and ships an interactive site that documents the work with evidence — model weights, logged rollouts, and qualitative evaluation clips.

**Live site:** clone and run locally (see [Quick start](#quick-start))  
**Artifacts:** [huggingface.co/fbarbosa1](https://huggingface.co/fbarbosa1)

---

## Summary

| Deliverable | Description |
|-------------|-------------|
| **Datasets** | Three LeRobot-format SO-101 teleop datasets + one logged eval rollout |
| **Models** | Four finetuned policies (SmolVLA, π₀.5 manipulation, ACT, π₀.5 driving) |
| **Hardware eval** | Closed-loop SO-101 rollouts, including published failure cases |
| **Driving eval** | Open-loop π₀.5 inference on held-out NVIDIA AV clips |
| **Interactive site** | [Explorer](explorer.html), [Methods](methods.html), [Model Zoo](models.html), [Driving](driving.html) |

---

## Problem

VLAs are hard to understand from papers and checkpoints alone. It is unclear what each part of the stack does — vision tokens, language conditioning, flow-matching action heads, action chunking — and why a policy can look fine in notebooks but fail on hardware.

**Hypothesis:** The same VLA recipe spans embodiments (arm vs. car): instruction + image + state → action chunk. If the pipeline can be explained once and validated with real training runs, the same framework applies to manipulation and driving.

---

## Approach

1. **Collect data** on an SO-101 follower arm using LeRobot conventions.
2. **Finetune** SmolVLA, π₀.5, and ACT baselines on teleop demonstrations.
3. **Evaluate** with closed-loop hardware rollouts and open-loop diagnostics; log failures to Hugging Face.
4. **Extend** the π₀.5 stack to NVIDIA PhysicalAI driving data and evaluate open-loop on held-out clips.
5. **Document** results in this repo and on the interactive site, prioritizing episode-level evidence over homepage benchmark tables.

---

## Technical Work

### Datasets (LeRobot format on Hugging Face)

| Dataset | Description |
|---------|-------------|
| [so101_embodied_tasks_v1](https://huggingface.co/datasets/fbarbosa1/so101_embodied_tasks_v1) | ~100 teleop episodes on SO-101 — fork pick-up and lego-into-cup tasks, top-down + wrist cameras |
| [so101_lego_into_glass](https://huggingface.co/datasets/fbarbosa1/so101_lego_into_glass) | Teleop demos for lego-into-glass (ACT baseline) |
| [eval_act_so101_lego_into_glass](https://huggingface.co/datasets/fbarbosa1/eval_act_so101_lego_into_glass) | Logged closed-loop policy rollout — arm misses the glass (failure evidence on site) |

Collected and exported with [LeRobot](https://github.com/huggingface/lerobot) (parquet + video, v3 schema).

### Models

| Model | Base | Method | Task |
|-------|------|--------|------|
| [smolvla_so101_embodied_v1](https://huggingface.co/fbarbosa1/smolvla_so101_embodied_v1) | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | BC finetune, frozen vision | SO-101 embodied tasks |
| [pi05_so101_embodied_v1](https://huggingface.co/fbarbosa1/pi05_so101_embodied_v1) | [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | BC finetune, flow-matching action expert | SO-101 embodied tasks |
| [act_so101_lego_into_glass_020000_v6](https://huggingface.co/fbarbosa1/act_so101_lego_into_glass_020000_v6) | — | Action Chunking Transformer | Lego into glass |
| [pi05-nvidia-av-generalize-012000](https://huggingface.co/fbarbosa1/pi05-nvidia-av-generalize-012000) | π₀.5 stack | BC on NVIDIA PhysicalAI AV clips | Ego-frame driving (open-loop eval) |

**Training stack:** LeRobot (`lerobot-train`), Modal GPU jobs for longer runs, local SO-101 eval with remote inference servers.

**Methods used:**

- Behavior cloning on teleop data with frozen vision encoders
- LoRA / action-expert-only training where full finetunes were too heavy
- Flow-matching action heads (π₀.5) — denoising an action chunk conditioned on tokens
- Action chunking (ACT) — predict H steps, execute, re-infer
- Open-loop diagnostics — replay training frames through a checkpoint to separate policy errors from deployment errors
- Closed-loop logging — record failed (and partial) real-robot episodes back to Hugging Face

See [methods.html](methods.html) for the finetuning guide (GRPO/DPO covered as general post-training concepts, not part of this driving work).

### Interactive site

| Page | Role |
|------|------|
| [Explorer](explorer.html) | Architecture-faithful teaching sim — patch tokens, action queries, flow τ, chunk heatmap, SO-101 motion (does not load 4B weights in-browser) |
| [Methods](methods.html) | Finetuning decisions and training recipe |
| [Models](models.html) | Model zoo + failure gallery with Hugging Face clips |
| [Driving](driving.html) | π₀.5 AV finetune + held-out open-loop clip gallery |
| [Home](index.html) | Project overview and artifact links |

---

## Results & Evidence

| Claim | Evidence |
|-------|----------|
| Trained real policies | Public Hugging Face checkpoints + training configs |
| Policies run on hardware | SO-101 rollouts; eval dataset with video |
| Policies do not always work | Failure examples published intentionally |
| Architecture understood | Explorer + methods text tied to trained models |
| Driving closed-loop | **Not claimed** — open-loop clip examples only |

### Failure analysis

- **ACT lego:** logged eval — policy never completes insertion ([video on site](models.html#fail-act))
- **SmolVLA:** partial fork/lego success; undershoot and placement sensitivity
- **π₀.5 manipulation:** language-conditioned; task bleed when scene is ambiguous
- **Driving:** qualitative open-loop overlays on held-out AV footage (ADE 0.42 m on 50-clip finetune)

### Limitations

- Explorer internals are illustrative, not attention maps from trained checkpoints
- No on-vehicle closed-loop deployment in this project
- No LLM fine-tuning or LLM-specific information (could be an extension)

---

## Quick start

```bash
git clone https://github.com/fbarbosa-stanford/vla-lab.git
cd vla-lab
python3 -m http.server 8124
# open http://localhost:8124
```

To reproduce training, use LeRobot with the linked Hugging Face datasets and base checkpoints. Hardware eval requires an SO-101 arm and the inference setup described on each model card.

---

## Repo structure

```
vla-lab/
├── index.html          # Project homepage
├── explorer.html       # Interactive VLA Explorer
├── methods.html        # Finetuning guide
├── models.html         # Model zoo + failure examples
├── driving.html        # Driving evaluation + clip gallery
├── css/  js/  vendor/  # Site assets
└── assets/driving/     # Open-loop driving preview clips
```

---

## Acknowledgments

| Resource | Role |
|----------|------|
| [LeRobot](https://github.com/huggingface/lerobot) | Dataset schema, training CLI, policy implementations |
| [SmolVLA](https://huggingface.co/papers/2506.01844) / [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | Compact VLA baseline |
| [Physical Intelligence π₀ / π₀.5](https://www.physicalintelligence.company/blog/pi05) / [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | Flow-matching VLA stack |
| [ALOHA](https://tonyzhaozh.github.io/aloha/) (Tony Zhao et al.) | Teleop / bimanual manipulation lineage |
| [Three.js](https://threejs.org/) | Explorer 3D scene (vendored) |
| NVIDIA PhysicalAI AV data | Driving finetuning source |
| SO-101 follower arm | Manipulation hardware |

---

## AI disclosure

AI-assisted tooling was used for LeRobot config debugging, site layout, model training, Explorer step animation, and ASCII diagrams across the site. All training decisions, eval logging, and claims about what is real vs. simulated were made and reviewed by Felipe Barbosa. This repository is authored by Felipe Barbosa only.

---

## License

Model weights on Hugging Face use the license stated on each model card (Apache-2.0 where applicable).
