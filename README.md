# The Realignment Problem: When Right becomes Wrong in LLMs

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
python datagen/oralce_triage_relabel.py \
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