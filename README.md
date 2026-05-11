# Unbiased Bytes: Evaluating Gender and Racial Bias in Open-Weight Instruction-Tuned LLMs

**DSCI 531 — Fairness in AI | University of Southern California | Spring 2026**

**Team:** Priya Prasad · Sapnil Patel · Vraj Patel

---

## Overview

This project evaluates gender and racial bias in three open-weight instruction-tuned language models using established fairness benchmarks. We measure both associative bias (stereotypical word associations) and reasoning bias (stereotypical inference under ambiguity), test two inference-time mitigation strategies, and analyze sensitivity to decoding stochasticity.

### Models Evaluated
| Model | Parameters | Source |
|-------|-----------|--------|
| LLaMA-3-8B-Instruct | 8B | Meta |
| Mistral-7B-Instruct-v0.2 | 7B | Mistral AI |
| Gemma-2-9B-IT | 9B | Google |

### Benchmarks
| Benchmark | Type | What It Measures |
|-----------|------|-----------------|
| [StereoSet](https://huggingface.co/datasets/McGill-NLP/stereoset) | Associative | Stereotype preference in sentence continuations |
| [CrowS-Pairs](https://huggingface.co/datasets/nyu-mll/crows_pairs) | Associative | Stereotype preference in minimal-edit sentence pairs |
| [BBQ](https://huggingface.co/datasets/heegyu/bbq) | Reasoning | Bias in question answering under ambiguity |

### Research Questions
1. **RQ1:** How do these models differ in gender and racial bias across benchmarks?
2. **RQ2:** How sensitive are bias metrics to decoding stochasticity?
3. **RQ3:** Do neutral framing and self-debiasing (Schick et al., 2021) reduce bias?

---

## Key Findings

- **Benchmark divergence:** All models show anti-stereotype preference on StereoSet (SPR 17–45%) but pro-stereotype preference on CrowS-Pairs (SPR 55–71%), suggesting RLHF alignment suppresses contextual stereotypes but not unconditional sentence-level likelihoods.
- **Self-debiasing works:** Contrastive logit suppression reduced Mistral's CrowS-Pairs SPR from 69% to 53% (p = 0.002, −16 pp).
- **Stochasticity matters:** Generation-based scoring produces significantly lower stereotype preference than likelihood-based scoring across all models (drops of 16–46 pp).
- **Overconfidence under ambiguity:** All models show high BBQ bias rates (44–68%) with very low unknown selection rates (0–22%).

---

## Repository Structure

```
UnbiasedBytes/
├── Unbiased_Bytes_Final_Pipeline.ipynb   # Main evaluation notebook (run on Colab)
├── outputs/                              # All generated results
│   ├── all_item_results.jsonl            # Item-level results (2,700 scored items)
│   ├── summary_all_models.csv            # Aggregated metrics by model/benchmark/bias
│   ├── cross_model_tests.csv             # Pairwise Mann-Whitney U tests
│   ├── demographic_disparity_tests.csv   # Gender vs race disparity tests
│   ├── intersectional_analysis.csv       # Intersectional group breakdowns
│   ├── mitigation_effectiveness.csv      # Baseline vs mitigation (Wilcoxon)
│   ├── stochasticity_results.csv         # Deterministic vs stochastic scoring
│   └── charts/                           # Publication-ready figures
│       ├── bar_chart_key_metrics.png
│       ├── heatmap_cross_model.png
│       ├── mitigation_effectiveness.png
│       └── stochasticity_effect.png
├── paper/
│   └── main.tex                          # Final paper (ACM sigconf format)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Google Colab Pro (A100 GPU recommended, High RAM enabled)
- Python 3.10+
- HuggingFace account with access to gated models (accept both licenses):
  - [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - [Gemma-2-9B-IT](https://huggingface.co/google/gemma-2-9b-it)

### Running the Pipeline

1. **Open the notebook in Colab:**

   Upload `Unbiased_Bytes_Final_Pipeline.ipynb` to Google Colab.

2. **Set the runtime:**

   Go to `Runtime → Change runtime type → A100 GPU`, and toggle **High RAM** on.

3. **Add your HuggingFace token:**

   Go to `Settings (gear icon) → Secrets → Add new secret`:
   - Name: `HF_TOKEN`
   - Value: your HuggingFace token (from https://huggingface.co/settings/tokens)

4. **Run all cells:**

   `Runtime → Run all`. The pipeline takes approximately 2–3 hours and processes models sequentially to manage GPU memory.

5. **Download results:**

   The final cell auto-zips all outputs and triggers a download.

### Running Locally (Advanced)

```bash
git clone https://github.com/VRAJ2202/UnbiasedBytes.git
cd UnbiasedBytes
pip install -r requirements.txt
```

Then open the notebook in Jupyter and ensure you have a CUDA GPU with ≥16 GB VRAM.

---

## Methodology

### Scoring
- **StereoSet & CrowS-Pairs:** Normalized per-token log-likelihood comparison
- **BBQ:** Log-likelihood scoring of answer options, prediction via argmax

### Mitigation Strategies
| Strategy | Mechanism |
|----------|-----------|
| Neutral Framing | Prepend fairness instruction to context |
| Self-Debiasing | Contrastive logit suppression (Schick et al., 2021): compute normal and bias-induced logits, suppress tokens amplified under biased framing |

### Statistical Tests
- **Cross-model:** Mann-Whitney U with Cohen's d effect sizes
- **Mitigation:** Wilcoxon signed-rank (paired by item)
- **Confidence intervals:** Bootstrap, 5,000 resamples, 95% level

### Configuration
| Parameter | Value |
|-----------|-------|
| Sample size | 100 items per benchmark per model |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Random seed | 42 |
| Stochastic samples | 10 per item (temp=0.7, top-p=0.9) |
| Total scored items | 2,700 (3 models × 3 benchmarks × 3 mitigations × 100) |

---

## Results Summary

### StereoSet SPR (%) — Baseline
| Model | Gender | Race |
|-------|--------|------|
| LLaMA-3-8B | 33.3 | 45.1 |
| Mistral-7B | 16.7 | 42.7 |
| Gemma-2-9B | 22.2 | 43.9 |

*All values below 50% → anti-stereotype preference*

### CrowS-Pairs SPR (%) — Baseline
| Model | Gender | Race |
|-------|--------|------|
| LLaMA-3-8B | 54.8 | 55.1 |
| Mistral-7B | 64.5 | 71.0 |
| Gemma-2-9B | 54.8 | 63.8 |

*All values above 50% → stereotype preference*

### Mitigation Highlight
| Strategy | Model | Benchmark | Baseline → Mitigated | p-value |
|----------|-------|-----------|---------------------|---------|
| Self-Debiasing | Mistral-7B | CrowS-Pairs | 69% → 53% | 0.002** |
| Neutral Framing | LLaMA-3-8B | StereoSet | 43% → 36% | 0.039* |

---

## Citation

```bibtex
@article{unbiasedbytes2026,
  title={Evaluating Gender and Racial Bias in Open-Weight Instruction-Tuned Large Language Models},
  author={Prasad, Priya and Patel, Sapnil and Patel, Vraj},
  journal={DSCI 531 Final Project, University of Southern California},
  year={2026}
}
```

---

## References

1. Nadeem et al. (2021). *StereoSet: Measuring stereotypical bias in pretrained language models.* ACL.
2. Nangia et al. (2020). *CrowS-Pairs: A challenge dataset for measuring social biases in masked language models.* EMNLP.
3. Parrish et al. (2022). *BBQ: A hand-built bias benchmark for question answering.* Findings of ACL.
4. Schick et al. (2021). *Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in NLP.* TACL.
5. Shaikh et al. (2023). *On second thought, let's not think step by step! Bias and toxicity in zero-shot reasoning.* ACL.

---

## License

This project is for academic purposes (DSCI 531, USC). Code is provided as-is for reproducibility.
