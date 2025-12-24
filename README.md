# XNet-DT: Direction-Aware and Resource-Balanced Dual-Tree Complex Wavelet Network

**Paper Title:** Direction-Aware and Resource-Balanced Dual-Tree Complex Wavelet Network for Fully- and Semi-Supervised Coronary Artery Segmentation

## Abstract
To address the challenges posed by intrinsic coronary artery characteristics and the scarcity of annotated medical imaging data, we propose **XNet-DT** – a novel and generalized architecture that seamlessly adapts to both fully-supervised and semi-supervised learning modes. XNet-DT utilizes the Dual-Tree Complex Wavelet Transform (DT-CWT) to balance high and low frequency branching in its design. Specifically, DT-CWT enhances vascular edge perception through six directional sub-bands (±15°, ±45°, ±75°) and dynamically allocates computational resources proportional to each sub-band’s energy content, thereby eliminating redundant computations.

We further introduce a composite loss function based on coronary centerline information specifically designed for coronary structures, which significantly improves the edge detection and bifurcation continuity. Extensive experiments conducted on the **ASOCA dataset** (the most widely used dataset in this field) show that XNet-DT is able to achieve a Dice score of **82.52%** in a fully-supervised environment and **78.63%** using only 25% of the labeled data, out-performing XNetv2 by **1.3%** for the same number of parameters. These results highlight its efficiency and robustness, paving the way for clinical applications in stenosis quantification and surgical planning.

---

## Model Architecture
<img width="346" alt="XNet-DT Architecture Diagram" src="https://github.com/user-attachments/assets/adc0d22c-3f27-4750-b73e-b81aaa09ff0a" />

*Overview of the XNet-DT framework, integrating DT-CWT for directional feature extraction and resource-aware computation.*

---

## Dataset
This work utilizes the **ASOCA (Automated Segmentation of Coronary Arteries)** dataset, the most widely adopted benchmark for coronary artery segmentation tasks. The dataset comprises challenging coronary CTA scans with corresponding expert annotations.

---

## Experimental Results

### Quantitative Comparison
<img width="636" alt="Quantitative Results Table" src="https://github.com/user-attachments/assets/919216f5-8955-4a28-98e0-65471cde0aae" />
*Comprehensive performance comparison (Dice, 95% Hausdorff Distance, Jaccard Index, ClDice) against state-of-the-art methods on the ASOCA dataset.*

### Ablation Studies

#### **TABLE II. Ablation Study on DT-CWT Basis**
| Wavelet Basis     | Dice   | 95HD  | JI    | ClDice |
|-------------------|--------|-------|-------|--------|
| Antonini          | 81.24  | 21.41 | 68.86 | 82.88  |
| LeGall            | 81.63  | 12.48 | 69.52 | 83.28  |
| **Near Sym A**    | **82.52** | **13.97** | **70.61** | **84.61** |
| Near Sym B        | 81.94  | 15.99 | 69.87 | 82.88  |
| Near Sym B Bp     | 81.92  | 14.79 | 69.80 | 83.19  |

#### **TABLE III. Ablation Study on Loss Function**
| Dice Loss | CE Loss | CBDice Loss | Dice   | 95HD  | JI    | ClDice |
|-----------|---------|-------------|--------|-------|-------|--------|
| ✓         |         |             | 81.61  | 15.96 | 69.34 | 82.85  |
|           | ✓       |             | 81.72  | 14.17 | 69.62 | 82.02  |
|           |         | ✓           | 81.49  | 18.58 | 69.16 | 83.01  |
| ✓         | ✓       | ✓           | **82.52** | **13.97** | **70.61** | **84.61** |

#### **TABLE IV. Impact of Skip Connections and Feature Fusion Strategies**
| Skip Connections | Fusion Strategies | Dice   | 95HD  | JI    | ClDice |
|------------------|-------------------|--------|-------|-------|--------|
|                  | CAT               | 81.77  | 17.47 | 69.57 | 82.33  |
|                  | ADD               | 81.90  | 17.83 | 69.67 | 83.76  |
| ✓                | CAT               | 81.80  | 16.71 | 69.64 | 82.97  |
| ✓                | ADD               | **82.52** | **13.97** | **70.61** | **84.61** |

### Qualitative Comparison
<img width="346" alt="Qualitative Segmentation Results" src="https://github.com/user-attachments/assets/2f758c0d-0f33-40b5-9355-63089a0157bd" />

*Qualitative results. To more objectively and efficiently validate the performance of our method, we selected representative segmentation regions from each dataset. Green arrows indicate areas of segmentation discontinuity, while yellow arrows highlight redundant blood vessels.*

---

## How to Use
*Instructions for setup, training, and inference will be provided here.*

---
*For more details, please refer to the full paper.*
