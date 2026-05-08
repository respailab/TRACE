# [The Realignment Problem: When Right becomes Wrong in LLMs](https://respailab.github.io/TRACE/)

**Accepted to ICML 2026 Main Track (26% acceptance rate)**

---

### Authors

**Aakash Sen Sharma**¹, **Debdeep Sanyal**², **Manodeep Ray**³, **Vivek Srivastava**³, **Shirish Karande**³, **Murari Mandal**⁴

¹ InvideoAI &nbsp;·&nbsp; ² Birla AI Labs &nbsp;·&nbsp; ³ TCS Research &nbsp;·&nbsp; ⁴ Kalinga Institute of Industrial Technology, Bhubaneswar

📬 Correspondence: Aakash Sen Sharma — aakash.sensharma@invideo.io

---

## Abstract

Post-training alignment of large language models (LLMs) relies on large-scale human annotations guided by policy specifications that change over time. Cultural shifts, value reinterpretations, and regulatory or industrial updates make static alignment increasingly brittle. As policies evolve, deployed models can diverge from current alignment objectives, creating an **Alignment–Reality Gap** that is difficult to audit or correct.

We introduce **TRACE** (Triage and Re-align by Alignment Conflict Evaluation), a framework that transforms re-alignment into a structured optimization problem over existing data **without requiring fresh human annotation**. Leveraging a stronger model as a proxy judge, TRACE operates via a three-stage pipeline: (1) triaging preference pairs into inversion, suppression, or retention categories based on alignment conflicts; (2) computing an alignment impact score via bi-level optimization to prioritize high-leverage samples; and (3) executing updates using a hybrid objective that combines relational losses (e.g., IPO) for preference inversion and punitive losses (e.g., NPO) for response suppression, anchored by a KL regularizer to preserve general capabilities.

Experiments on **Qwen2.5-7B**, **Gemma-2-9B**, and **Llama-3.1-8B** demonstrate robust re-alignment on synthetic benchmarks and the PKU-SafeRLHF dataset without degrading general utility. TRACE achieves an **81.8% human preference win rate** over the U2A unlearning baseline while preserving MMLU and GSM8K performance within confidence intervals.

---

## Datasets

**Step 1 — Build the initial preference dataset:**

```bash
python datagen/create_pku_dpo_dataset.py \
    --output_json pku_base.json \
    --num_samples 20000
```

This pulls from PKU-SafeRLHF and produces `{prompt, chosen, rejected}` triples.

**Step 2 — Oracle triage + DPO-Gold relabelling:**

```bash
python datagen/oracle_triage_relabel.py \
    --input_json pku_base.json \
    --output_json pku_dpo_gold.json \
    --oracle_model gpt-4o \
    --ckpt_every 1000
```

For each pair, GPT-4o checks both responses against the new policy (π_new) and relabels:
- `chosen` violates, `rejected` does not → **INVERT**: swap chosen/rejected
- both violate → **PUNISH**: keep pair, suppress both during training
- `chosen` does not violate → **RETAIN**: keep as-is

This produces `pku_dpo_gold.json` (relabelled pairs with `category` field) and `pku_dpo_gold_triage.json` (triage-only view). Checkpoints are saved to `dpo_gold_checkpoints/` every `--ckpt_every` samples so the run can be resumed if interrupted.

**Step 3 — Stratified train/test split:**

```bash
python datagen/prepare_datasets_stratified.py \
    --ckpt_json pku_dpo_gold.json \
    --output_dir splits \
    --train1_size 10000 \
    --train2_size 5000
```

This stratifies by INVERT/PUNISH/RETAIN category so the distribution is consistent across splits. Outputs two formats under `splits/`:
- `splits/trace/` — full records with `category` field (used by TRACE)
- `splits/dpo/` — `{prompt, chosen, rejected}` only (used by DPO baselines)

---

## Usage

###  Run TRACE Re-alignment

```bash
accelerate launch src/trace.py \
    --model_name_or_path 'meta-llama/Llama-3.1-8B' \
    --dataset_json_path splits/trace/train1.json \
    --output_dir runs/trace \
    --beta 0.3 --alpha_kl 0.1 \
    --gold_batch_size 100 --pa_batch_size 2 \
    --batch_size 1 --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 --num_train_epochs 1 \
    --lamda 1.0
```

To run without LoRA adapters (full fine-tune):
```bash
accelerate launch src/trace.py ... # omit --use_lora (default True uses LoRA)
```


### Ablations

| Variant | Script | Key flag |
|---|---|---|
| Oracle-guided PUNISH | `trace_oracle.py` | `--oracle_max_tokens 512` |
| No impact weighting | `trace_no_impact_score.py` | — (uniform weights baked in) |
| No triage | `trace_notriage.py` | — (NPO on all samples) |
| No KL regularization | `trace.py` | `--alpha_kl 0.0` |


---

## 📊 Results

### Human Preference Evaluation (Table 2)

Annotators were shown triplets of responses (DPO-Gold, TRACE, U2A) and selected the one best following π_new, with no knowledge of model identity. TRACE substantially closes the gap between purely punitive unlearning and gold-standard full re-annotation.

| Method | PKU-SafeRLHF Win Rate vs. U2A | SynthValueBench Win Rate vs. U2A | Krippendorff's α |
|---|---|---|---|
| DPO-Gold (upper bound) | 87.1% | 92.4% | 0.80 |
| **TRACE (Ours)** | **81.8%** | **85.3%** | **0.77** |
| U2A (baseline) | — | — | 0.76 |

### General Capability Benchmarks (Table 3, PKU-SafeRLHF)

TRACE preserves MMLU and GSM8K within confidence intervals of the base model. The HellaSwag reduction (~3.2 points) is bounded and favorable given the 81.8% policy adherence gain over U2A.

| Method | GPQA ↑ | MMLU ↑ | HellaSwag ↑ | GSM8K ↑ |
|---|---|---|---|---|
| Base Model | 31.6 ± 0.9 | **70.6 ± 0.8** | **81.4 ± 1.0** | 70.4 ± 0.8 |
| DPO-Gold | **32.1 ± 1.1** | 70.5 ± 0.9 | 81.3 ± 1.2 | **70.8 ± 1.0** |
| **TRACE (Ours)** | 30.1 ± 0.1 | 70.2 ± 0.8 | 78.2 ± 0.9 | 70.6 ± 0.7 |
| U2A | 29.5 ± 0.3 | 70.2 ± 1.1 | 80.8 ± 1.2 | 69.9 ± 1.1 |

### Target Policy Agreement (Table 5, PKU-SafeRLHF)

Target Policy Agreement (TPA) measures the percentage of model responses that comply with π_new, assessed by the policy oracle. Results shown at varying data scales across two model families, confirming that TRACE's gains come from triage and impact weighting — not oracle access alone.

| Model | Data Size | Naive Oracle DPO | **TRACE (Ours)** | Δ |
|---|---|---|---|---|
| Llama-3.1-8B | 5k | 35.9 | **55.9** | +20.0 |
| Llama-3.1-8B | 10k | 46.5 | **65.5** | +19.0 |
| Llama-3.1-8B | 20k | 52.2 | **70.7** | +18.5 |
| Gemma-2-9B | 5k | 37.1 | **54.4** | +17.3 |
| Gemma-2-9B | 10k | 48.2 | **66.9** | +18.7 |
| Gemma-2-9B | 20k | 53.9 | **71.8** | +17.9 |

TRACE at 5k samples outperforms Naive Oracle DPO at 20k samples, isolating the contribution of the triage and impact-weighting mechanism from oracle relabeling.

### Adversarial Robustness — Attack Success Rate (Table 6, lower is better)

Two adversarial attack strategies: **Fictional Scenario Nesting** (wraps harmful queries inside creative writing) and **Refusal Suppression** (appends explicit anti-refusal constraints). Results averaged across model architectures.

| Method | Fictional Scenario Nesting ↓ | Refusal Suppression ↓ |
|---|---|---|
| DPO-Gold (upper bound) | **11.3 ± 0.7** | **12.8 ± 1.1** |
| **TRACE (Ours)** | 27.3 ± 1.2 | **19.7 ± 1.0** |
| U2A (baseline) | 24.6 ± 0.8 | 21.3 ± 1.3 |

### Oracle Noise Robustness (Table 8)

TRACE degrades gracefully under label noise. Even with 20% flipped triage labels, TRACE outperforms U2A with a clean oracle.

| Oracle Noise Level | DPO-Gold | U2A | **TRACE (Ours)** |
|---|---|---|---|
| 0% (No Noise) | 73.8% | 55.8% | **70.2%** |
| 10% Flipped Labels | 69.9% | 55.3% | **67.4%** |
| 20% Flipped Labels | 62.4% | 54.7% | **61.5%** |

### Component-wise Ablation (Table 4, Llama-3.1-8B, PKU-SafeRLHF)

| Variant | Policy Agreement ↑ | MMLU ↑ | ASR ↓ |
|---|---|---|---|
| **TRACE (Full)** | **70.7** | **70.2** | **27.3** |
| w/o Triage | 58.1 | 70.2 | 24.6 |
| w/o KL Regularization | 71.5 | 64.1 | 29.8 |
| w/o Impact Weighting | 62.8 | 69.5 | 32.1 |

---

## 📁 Repository Structure

```
TRACE/
├── merge_lora_adapters.py             # Merge LoRA adapters into base model
├── accel_config.yaml                  # Accelerate multi-GPU config (3× A6000)
│
├── src/                               # TRACE core (Stages 2–3)
│   ├── trace.py                  # Main: impact scoring + hybrid update (INVERT/PUNISH/RETAIN)
│   ├── trace_oracle.py           # Main + oracle-guided DPO for PUNISH branch
│   ├── trace_no_impact_score.py       # Ablation: uniform impact weights (no Phase 2)
│   ├── trace_notriage.py              # Ablation: NPO on all samples (no triage)
│   └── utils/
│       ├── common.py                  # conjugate_gradient(), hvp(), trainable_params()
│       ├── reward.py                  # preference_loss(), LossConfig, _get_batch_logps()
│       ├── losses.py                  # Impact score computation (compute_impact_scores, gamma damping)
│       ├── datamodule.py              # Dataset loading and batch collation
│       └── oracle.py                  # GPT-4o oracle correction for PUNISH samples
│
├── evaluation/
│   ├── generate_general.py            # Generate responses: GPQA, MMLU, HellaSwag, GSM8K
│   ├── generate_tpa.py                # Generate responses for TPA benchmark
│   ├── generate_tpa_vllm.py           # TPA generation with vLLM (faster)
│   ├── oracle_evaluate_general.py     # GPT-4o scoring for capability benchmarks
│   ├── oracle_evaluate_tpa.py         # GPT-4o scoring for TPA
│   └── generate_adversarial_eval.py   # Generate responses for ASR evaluation (Fictional Nesting, Refusal Suppression)
│
├── datagen/
│   ├── prepare_datasets.py            # Train/test split (uniform)
│   ├── prepare_datasets_stratified.py # Stratified split by INVERT/PUNISH/RETAIN
│   ├── create_pku_dpo_dataset.py      # Step 1: Pull PKU-SafeRLHF → {prompt, chosen, rejected}
│   ├── oracle_triage_relabel.py       # Step 2: GPT-4o triage + DPO-Gold relabelling
│   ├── prepare_datasets_stratified.py # Step 3: Stratified train/test split by category 
│   └── create_noisy_splits.py         # Inject label noise for robustness ablations
│
├── ablations/
│   ├── dpo_accelerate.py              # Standard DPO baseline
    └── robust_dpo_accelerate.py       # rDPO baseline (Chowdhury et al.)
```

---

## Citation

```bibtex
@inproceedings{sensharma2026realignment,
  title={The Realignment Problem: When Right becomes Wrong in LLMs},
  author={Sen Sharma, Aakash and Sanyal, Debdeep and Ray, Manodeep and Srivastava, Vivek and Karande, Shirish and Mandal, Murari},
  booktitle={International Conference on Machine Learning},
  year={2026},
  organization={PMLR}
}
```

---

## 📬 Contact

For questions about the paper or code:

- **Aakash Sen Sharma** → [aakash.sensharma@invideo.io](mailto:aakash.sensharma@invideo.io)
- **Debdeep Sanyal** → [debdeep.respailab@gmail.com](mailto:debdeep.respailab@gmail.com)
- **Manodeep Ray** → [manodeepray1@gmail.com](mailto:manodeepray1@gmail.com)
- **Vivek Srivastava** → [vivek.srivastava@tcs.com](mailto:vivek.srivastava@tcs.com)
- **Shirish Karande** → [shirish.karande@tcs.com](mailto:shirish.karande@tcs.com)
- **Murari Mandal** → [murari.mandalfcs@kiit.ac.in](mailto:murari.mandalfcs@kiit.ac.in)

Alternatively, open a **GitHub Issue** in this repository.
