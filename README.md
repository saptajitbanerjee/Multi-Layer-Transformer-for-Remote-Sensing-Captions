# Multi-Layer Transformer for Remote Sensing Captions

A deep learning image captioning system for satellite imagery, built around a custom Multi-Layer Transformer with an EfficientNetB7 backbone. Evaluated against a single-layer baseline on a domain-specific satellite dataset, and stress-tested zero-shot on independent, unannotated real-world conflict-zone imagery.

🔗 **[Multi-Layer Transformer on Kaggle](https://www.kaggle.com/code/rogaldorn7/multi-layer-transformer-remote-sensing-captions)** &nbsp;|&nbsp; **[Single-Layer Transformer on Kaggle](https://www.kaggle.com/code/rogaldorn7/single-layer-transformer-remote-sensing-captions)**


> **Scope note:** This repo covers my individual contribution to a 3-person team project — architecture design and optimization of the Single- and Multi-Layer Transformer pipelines, training and evaluation of both models on the satellite dataset, and zero-shot real-world validation. Teammates Umika Khugsal and Aritra Das contributed the Phase 1 recurrent baselines (CNN-LSTM/CNN-GRU), the data/tokenization pipeline, and the augmentation framework — full credits and methodology are in the [project report](./report/Project_Report.pdf).

## Key Result

The proposed **Multi-Layer Transformer** (EfficientNetB7 backbone, 3 encoder + 3 decoder blocks, 8 attention heads, SiLU activations, custom label smoothing) decisively outperformed the single-layer baseline across every metric:

| Metric | Single Layer (K=2) | Multi Layer (K=1) | Improvement |
|---|---|---|---|
| Accuracy | 40.21% | **45.43%** | **+13.0%** |
| Loss | 15.3162 | **3.6234** | **−76.3%** |
| BLEU-1 | 47.26% | **58.24%** | +23.2% |
| BLEU-2 | 27.86% | **41.07%** | +47.4% |
| BLEU-3 | 18.01% | **30.96%** | +71.1% |
| BLEU-4 | 11.46% | **23.83%** | **+107.9%** |

BLEU-4 — the strictest metric, requiring correct 4-word sequence matches — **more than doubled**, indicating genuine gains in contextual comprehension rather than isolated word-level guessing.

## The Problem

Standard image captioning architectures are built for ground-level photography and struggle with satellite imagery's top-down perspective, extreme scale variation, and dense object clustering. Recurrent sequence models (CNN-LSTM, CNN-GRU) compound this with an information bottleneck that fails to retain context over longer captions. This project evaluates whether a deeper, domain-adapted Transformer architecture can close that gap.

## Methodology

**Phase 1 — Baseline selection (Flickr8K):** Three architectures — CNN-GRU, CNN-LSTM, and a Single Layer Transformer (frozen EfficientNetB0 backbone) — were benchmarked on Flickr8K to identify the strongest foundation.

| Model | Val. Accuracy | Val. Loss | Beam(K=2) BLEU-1 |
|---|---|---|---|
| CNN-LSTM | 38.84% | 3.2028 | 60.21% |
| CNN-GRU | 40.53% | 3.1306 | 59.88% |
| **Single Layer Transformer** | **41.39%** | 11.2521 | **60.58%** |

The Single Layer Transformer's Self-Attention and Cross-Attention mechanisms outperformed both recurrent baselines by directly capturing long-range image-text dependencies, and was selected for Phase 2.

**Phase 2 — Domain adaptation (Satellite Image Caption Generation dataset):** The Single Layer baseline was scaled into the proposed Multi-Layer Transformer:

![Multi Layer Transformer Architecture Diagram](./Multi_Layer_Transformer_Architecture%20Diagram.png)

Key architectural changes over the baseline:
- **Backbone upgrade**: EfficientNetB0 → **EfficientNetB7** for richer spatial feature extraction.
- **Depth**: 1 → **3 encoder and 3 decoder blocks**, 8 attention heads per block.
- **Activations**: ReLU → **SiLU (Swish)** in FFN sublayers, for improved gradient flow.
- **Regularization**: L2 (which empirically hurt performance) replaced with **Dropout (0.5)**.
- **Loss**: Standard cross-entropy → **custom Manual Label Smoothing**, to improve calibration and reduce overconfidence.
- **Training**: Adam optimizer with warmup LR scheduling, gradient clipping, and early stopping. Batch size reduced from 64 → 8 to resolve Kaggle GPU OOM errors from the parameter increase.
- **Augmentation**: Full 360° rotation, horizontal/vertical flipping, and contrast adjustment, to handle satellite imagery's arbitrary orientation.
- **Inference**: Length-normalized Beam Search evaluated at widths K ∈ {1, 2, 3, 4}.

## Results

Full BLEU-1 through BLEU-4 scores across all tested beam widths:

| Model | Beam Width | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|---|
| Single Layer | 1 | 48.85% | 28.06% | 17.85% | 11.37% |
| Single Layer | 2 | 47.26% | 27.86% | 18.01% | 11.46% |
| Single Layer | 3 | 45.33% | 26.38% | 16.66% | 10.21% |
| Single Layer | 4 | 44.97% | 26.35% | 16.53% | 9.94% |
| **Multi Layer** | **1** | **58.24%** | **41.07%** | **30.96%** | **23.83%** |
| Multi Layer | 2 | 57.29% | 40.50% | 30.39% | 23.23% |
| Multi Layer | 3 | 55.63% | 39.36% | 29.68% | 22.93% |
| Multi Layer | 4 | 53.86% | 38.05% | 28.82% | 22.39% |

A consistent pattern across both models: BLEU scores decline monotonically as beam width increases — the highest-confidence single beam (K=1) already captures the most semantically appropriate sequence, and wider search doesn't help in this setting. The decline is steeper for the Single Layer model, indicating it's more prone to error propagation over longer sequences, while the Multi Layer model's more gradual decline reflects genuinely coherent multi-word phrase generation.

Full raw performance data: **[Model Performance Report](./results/Model_Performance_Report.xlsx)**

## Code

- [Multi-Layer Transformer notebook](./notebooks/Multi_Layer_Transformer.ipynb) — the proposed architecture, training, and evaluation. Also live on [Kaggle](https://www.kaggle.com/code/rogaldorn7/multi-layer-transformer-remote-sensing-captions).
- [Single-Layer Transformer notebook](./notebooks/Single_Layer_Transformer.ipynb) — baseline model and evaluation. Also live on [Kaggle](https://www.kaggle.com/code/rogaldorn7/single-layer-transformer-remote-sensing-captions). Full results also detailed in [Single_Layer_Transformer_Results.pdf](./results/Single_Layer_Transformer_Results.pdf).

## Qualitative Comparison

On unseen imagery, the Multi-Layer model consistently resolves scene semantics that the Single Layer baseline misreads from superficial visual cues:

![Qualitative Comparison](./results/Qualitative%20Comparision.png)

## Zero-Shot Validation: Real-World Conflict-Zone Imagery

To test generalization beyond the training distribution, the Multi-Layer Transformer was evaluated zero-shot on independent, unannotated satellite imagery of the 2026 Iran War, sourced from Reuters and entirely excluded from training/validation. The model correctly identified domain-relevant elements — vessels, storage tanks, industrial infrastructure — despite no conflict-zone imagery appearing anywhere in training:

![Captioning the 2026 Iran War Conflict Zones](./results/Captioning%20The%202026%20IRAN%20WAR%20Conflict%20Zones.png)

This confirms the model's captioning ability generalizes to genuinely out-of-distribution, operationally sourced imagery rather than just held-out samples from the training distribution.

## Repository Structure

```
├── notebooks/
│   ├── Multi_Layer_Transformer.ipynb
│   └── Single_Layer_Transformer.ipynb
├── presentation/
│   ├── Final Project Presetation - ID5030.pdf
│   └── Final Project Presetation - ID5030.pptx
├── report/
│   └── Project_Report.pdf
├── results/
│   ├── Captioning The 2026 IRAN WAR Conflict Zones.png
│   ├── Model_Performance_Report.xlsx
│   ├── Qualitative Comparision.png
│   ├── Single_Layer_Transformer_Results.pdf
│   └── Multiple_Layer_Transformer_Results.pdf
├── LICENSE
├── Multi_Layer_Transformer_Architecture Diagram.png
├── README.md
└── Supplementary_File.pdf
```

## Datasets

- **[Flickr8K](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k)** — Hodosh, Young & Hockenmaier (2013) — used for Phase 1 baseline architecture selection.
- **[Satellite Image Caption Generation](https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation/data)** (Kaggle) — used for Phase 2 domain-specific training.

## Future Work

- Scale encoder/decoder depth and embedding dimension further, contingent on access to HPC resources beyond Kaggle's GPU memory limits.
- Replace standard Transformer attention with a **Swin Transformer** to reduce self-attention complexity from quadratic to linear for high-resolution imagery.
- Apply **LoRA**, Flash Attention, and model quantization for memory and inference efficiency.

## Report & Presentation

- 📄 [Full project report](./report/Project_Report.pdf) — complete methodology, architecture details, and results.
- 🖥️ [Presentation slides](<./presentation/Final Project Presetation - ID5030.pdf>) — condensed overview of the approach and findings.
- 📊 [Model Performance Report](./results/Model_Performance_Report.xlsx) — full raw benchmark data across all models, phases, and beam widths.
- 📎 [Supplementary file](./Supplementary_File.pdf) — additional material referenced in the report.

## Author

**Saptajit Banerjee** — Designed and developed the Single- and Multi-Layer Transformer pipelines (EfficientNet backbones, Batch Normalization, Positional Embedding, Multi-Head Self/Cross-Attention); introduced architectural optimizations (SiLU/GELU activations, custom Manual Label Smoothing loss, 360° augmentation strategy, Adam + LR scheduler + early stopping); trained and evaluated both models on the satellite dataset across Accuracy, Cross-Entropy Loss, and BLEU-1–4 at beam widths K ∈ {1,2,3,4}; sourced and validated zero-shot generalization on independent real-world satellite imagery.

Full team credits and individual contributions in the [project report](./report/Project_Report.pdf).

## License

MIT License — see [LICENSE](./LICENSE) for details.
