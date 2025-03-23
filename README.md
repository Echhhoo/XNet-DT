# XNet-DT
**Paper Title:**  Direction-Aware and Resource-Balanced Dual-Tree Complex Wavelet Network for Fully- and Semi-Supervised Coronary Artery Segmentation

**Abstract** To address the challenges posed by intrinsic coronary artery characteristics and the scarcity of annotated medical imaging data, we propose XNet-DT - a novel and generalized architecture that seamlessly adapts to both fully-supervised and semi-supervised learning modes. XNet-DT utilizes the Du-al-Tree Complex Wavelet Transform (DT-CWT) to balance high and low frequency branching in its design. Specifically, DT-CWT enhances vascular edge perception through six directional sub-bands (±15°, ±45°, ±75°) and dynamically allocates computational resources proportional to each sub-band’s energy content, thereby eliminating redundant computations. We further introduce a composite loss function based on coronary centerline information specifically designed for coronary structures, which signifi-cantly improves the edge detection and bifurcation continuity. Extensive experiments conducted on the ASOCA dataset (the most widely used da-taset in this field) show that XNet-DT is able to achieve a Dice score of 82.52% in a fully-supervised environment and 78.63% using only 25% of the labeled data, out-performing XNetv2 by 1.3% for the same number of parameters. These results highlight its efficiency and robustness, paving the way for clinical applications in stenosis quantification and surgical plan-ning.


# How to Use


# Model Architecture

<img width="346" alt="image" src="https://github.com/user-attachments/assets/adc0d22c-3f27-4750-b73e-b81aaa09ff0a" />

# Dataset

# Quantitative Comparison
<img width="636" alt="截屏2025-03-23 15 36 06" src="https://github.com/user-attachments/assets/919216f5-8955-4a28-98e0-65471cde0aae" />

# Qualitative  Comparison
<img width="346" alt="image" src="https://github.com/user-attachments/assets/2f758c0d-0f33-40b5-9355-63089a0157bd" />
Qualitative results. To more objectively and efficiently validate the performance of our method, we selected representative segmentation regions from each dataset. Green arrows indicate areas of segmentation discontinuity, while yellow arrows highlight redundant blood vessels.
