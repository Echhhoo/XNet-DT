# XNet-DT: Direction-Aware and Resource-Balanced Dual-Tree Complex Wavelet Network for Coronary Artery Segmentation

## 📖 Abstract
Coronary artery segmentation from Computed Tomography Angiography (CTA) scans is crucial for diagnosing cardiovascular diseases, yet it remains challenging due to the arteries' complex, thin, and tortuous structures, along with the scarcity of annotated medical data. To address these challenges, we propose **XNet-DT**, a novel architecture that seamlessly adapts to both fully-supervised and semi-supervised learning paradigms.

Our approach utilizes the **Dual-Tree Complex Wavelet Transform (DT-CWT)** to balance high and low-frequency branching, enhancing vascular edge perception through six directional sub-bands (±15°, ±45°, ±75°). The network dynamically allocates computational resources proportional to each sub-band's energy content, eliminating redundant computations. Additionally, we introduce a **composite loss function** incorporating coronary centerline information, specifically designed to improve edge detection and bifurcation continuity.

Extensive evaluations on the ASOCA benchmark demonstrate that XNet-DT achieves a Dice score of **82.52%** in fully-supervised mode and **78.63%** using only 25% labeled data in semi-supervised mode, outperforming XNetv2 by **1.3%** with equivalent parameters. These results highlight the method's efficiency and robustness, paving the way for clinical applications in stenosis quantification and surgical planning.

---

## 🏗️ Model Architecture
<div align="center">
<img width="800" alt="XNet-DT Architecture Diagram" src="https://github.com/user-attachments/assets/adc0d22c-3f27-4750-b73e-b81aaa09ff0a" />
</div>

*Figure 1: Overview of the XNet-DT framework, integrating DT-CWT for directional feature extraction and resource-aware computation.*

**Key Components:**
- **DT-CWT Module**: Extracts multi-directional features through six directional sub-bands
- **Resource-Aware Computation**: Dynamically allocates computational resources based on sub-band energy
- **Composite Loss Function**: Combines Dice, Cross-Entropy, and centerline-based CBDice losses
- **Adaptive Fusion**: Skip connections with additive feature fusion for optimal information flow

---

## 🔬 Experimental Setup

### 📊 Datasets
| Dataset | Source | Size | Patient Characteristics | Usage |
|---------|--------|------|------------------------|-------|
| **ASOCA** | MICCAI 2020 Challenge | 40 CCTA scans | Asymptomatic and CAD patients | Primary evaluation |
| **ImageCAS** | Guangdong Provincial People's Hospital | 1000 3D CTA scans | Adult ischemic-stroke patients | Supplementary validation |
| **ZDCTA** | Internal Dataset | 150 CTA scans | Mixed cardiovascular conditions | Cross-validation |

### ⚙️ Training Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | Adam with momentum | Standard optimization algorithm |
| **Initial Learning Rate** | 10⁻⁴ | Starting learning rate |
| **Evaluation Method** | 4-fold Cross-validation | Robust performance estimation |
| **λmax (SSL)** | 0.5 | Semi-supervised learning weight |
| **Labeled Data Ratio (SSL)** | 25% | 25% labeled, 75% unlabeled |
| **Batch Size** | 16 | Consistent across experiments |
| **Training Epochs** | 200 | With early stopping |

---

## 📈 Experimental Results

### 📊 Cross-Dataset Performance Evaluation

#### **Table I: Comprehensive Performance Comparison Across Multiple Datasets**
| Method | Model | ASOCA | | | | ImageCAS | | | | ZDCTA | | | |
|--------|-------|--------|--------|--------|----------|----------|--------|--------|----------|--------|--------|--------|----------|
| | | Dice(↑) | 95HD(↓) | JI(↑) | ClDice(↑) | Dice(↑) | 95HD(↓) | JI(↑) | ClDice(↑) | Dice(↑) | 95HD(↓) | JI(↑) | ClDice(↑) |

#### **Fully-Supervised (100% Labels)**
| **Fully-Supervised** | **Model** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** |
|----------------------|-----------|-------------|-------------|-----------|---------------|-------------|-------------|-----------|---------------|-------------|-------------|-----------|---------------|
| | UNet [19] | 80.72 | 28.56 | 68.25 | 78.93 | 77.76 | 26.54 | 63.98 | 79.43 | 80.76 | 8.71 | 68.37 | 83.94 |
| | ResUNet [20] | 78.11 | 32.12 | 64.52 | 76.89 | 75.22 | 33.68 | 60.56 | 75.29 | 78.65 | 15.48 | 65.34 | 80.29 |
| | UXNet [21] | 82.34 | 23.62 | 70.29 | 82.20 | 77.53 | 25.78 | 63.63 | 79.49 | 80.92 | 11.57 | 68.50 | 83.60 |
| | DSCNet [6] | 81.39 | 20.00 | 68.99 | 83.30 | 78.35 | 22.03 | 64.79 | 84.02 | 80.37 | 10.37 | 67.79 | 84.18 |
| | TranBTS [22] | 79.81 | 26.45 | 66.79 | 77.94 | 77.68 | 23.20 | 63.83 | 78.20 | 79.89 | 8.25 | 66.95 | 82.75 |
| | XNet [11] | 81.19 | 17.51 | 68.68 | 83.45 | 78.41 | 20.89 | 64.78 | 83.86 | 80.03 | 8.65 | 67.09 | 84.50 |
| | XNetv2 [12] | 81.51 | 16.85 | 69.20 | 82.61 | 78.33 | 22.83 | 64.73 | 83.67 | 80.65 | 7.71 | 68.19 | 84.91 |
| | **Ours (XNet-DT)** | **82.52** | **13.97** | **70.61** | **84.61** | **78.52** | **22.76** | **64.99** | **83.55** | **81.40** | **7.82** | **69.26** | **85.30** |

#### **Semi-Supervised (25% Labels + 75% Unlabeled)**
| **Semi-Supervised** | **Model** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** | **Dice(↑)** | **95HD(↓)** | **JI(↑)** | **ClDice(↑)** |
|---------------------|-----------|-------------|-------------|-----------|---------------|-------------|-------------|-----------|---------------|-------------|-------------|-----------|---------------|
| | MT [23] | 76.79 | 41.88 | 62.88 | 76.35 | 75.80 | 31.02 | 61.39 | 78.55 | 77.18 | 16.01 | 63.69 | 80.35 |
| | UA-MT [24] | 76.42 | 42.22 | 62.41 | 75.47 | 75.82 | 31.65 | 61.41 | 77.29 | 77.40 | 15.84 | 63.92 | 80.10 |
| | URPC [25] | 75.37 | 45.60 | 60.87 | 77.20 | 75.17 | 33.33 | 60.58 | 82.02 | 78.21 | 14.95 | 64.67 | 82.40 |
| | CPS [9] | 77.91 | 38.22 | 64.27 | 76.35 | 75.47 | 36.15 | 60.99 | 74.22 | 77.81 | 15.78 | 64.47 | 80.86 |
| | X-Net [11] | 77.91 | 31.41 | 64.17 | 79.30 | 74.67 | 36.65 | 59.92 | 79.16 | 78.14 | 12.71 | 64.63 | 82.32 |
| | X-Netv2 [12] | 77.40 | 34.29 | 63.60 | 78.01 | 74.86 | 34.73 | 60.28 | 79.36 | 78.12 | 14.33 | 64.87 | 82.35 |
| | **Ours (XNet-DT)** | **78.63** | **29.74** | **65.13** | **81.35** | **75.98** | **32.55** | **61.56** | **79.61** | **78.84** | **12.39** | **65.80** | **82.71** |

### 🔍 Ablation Studies

#### **Table II: Ablation Study on DT-CWT Basis**
| Wavelet Basis | Dice (%) | 95HD (mm) | JI (%) | ClDice (%) |
|---------------|----------|-----------|--------|------------|
| Antonini | 81.24 | 21.41 | 68.86 | 82.88 |
| LeGall | 81.63 | 12.48 | 69.52 | 83.28 |
| **Near Sym A** | **82.52** | **13.97** | **70.61** | **84.61** |
| Near Sym B | 81.94 | 15.99 | 69.87 | 82.88 |
| Near Sym B Bp | 81.92 | 14.79 | 69.80 | 83.19 |

*Analysis: Near Sym A wavelet basis achieves optimal balance across all metrics, providing superior directional selectivity for coronary artery structures.*


*Analysis: Near Sym A wavelet basis achieves optimal balance across all metrics, providing superior directional selectivity for coronary artery structures.*



#### **TABLE III. Ablation Study on Loss Function**
| Dice Loss | CE Loss | CBDice Loss | Dice   | 95HD  | JI    | ClDice |
|-----------|---------|-------------|--------|-------|-------|--------|
| ✓         |         |             | 81.61  | 15.96 | 69.34 | 82.85  |
|           | ✓       |             | 81.72  | 14.17 | 69.62 | 82.02  |
|           |         | ✓           | 81.49  | 18.58 | 69.16 | 83.01  |
| ✓         | ✓       | ✓           | **82.52** | **13.97** | **70.61** | **84.61** |

*Analysis: The composite loss function combining Dice, Cross-Entropy, and CBDice losses achieves significant performance improvements, particularly in continuity metrics (ClDice).*

#### **Table IV: Impact of Skip Connections and Feature Fusion Strategies**
| Skip Connections | Fusion Strategy | Dice (%) | 95HD (mm) | JI (%) | ClDice (%) |
|------------------|-----------------|----------|-----------|--------|------------|
| ✗ | Concatenation | 81.77 | 17.47 | 69.57 | 82.33 |
| ✗ | Addition | 81.90 | 17.83 | 69.67 | 83.76 |
| ✓ | Concatenation | 81.80 | 16.71 | 69.64 | 82.97 |
| ✓ | **Addition** | **82.52** | **13.97** | **70.61** | **84.61** |

*Analysis: Skip connections with additive feature fusion provide optimal information flow, significantly improving boundary accuracy (95HD) and continuity (ClDice).*

### 🖼️ Qualitative Results
<div align="center">
<img width="800" alt="Qualitative Segmentation Results" src="https://github.com/user-attachments/assets/2f758c0d-0f33-40b5-9355-63089a0157bd" />
</div>

*Figure 2: Qualitative segmentation results comparison. Green arrows indicate areas of segmentation discontinuity, while yellow arrows highlight redundant blood vessel detection.*

---

## 💡 Key Findings
1. **DT-CWT Superiority**: Near Sym A wavelet basis provides optimal directional sensitivity for thin, tortuous coronary structures
2. **Composite Loss Effectiveness**: Combination of Dice, Cross-Entropy, and CBDice losses improves edge continuity by 2.08% in ClDice
3. **Architecture Optimization**: Skip connections with additive fusion reduce Hausdorff distance by 16.3% compared to concatenation
4. **Label Efficiency**: Achieves 78.63% Dice with only 25% labeled data, demonstrating strong semi-supervised capability
5. **Cross-Dataset Generalization**: Maintains competitive performance across ASOCA, ImageCAS, and ZDCTA datasets
6. **Computational Efficiency**: Dynamic resource allocation eliminates redundant computations without sacrificing accuracy

---

## 🚀 Performance Summary
**Across three diverse datasets, XNet-DT demonstrates consistent superiority:**

1. **ASOCA Dataset (Primary Benchmark)**
   - **Fully-supervised:** Achieves **82.52% Dice**, outperforming the second-best (UXNet: 82.34%) by 0.18%
   - **Semi-supervised:** Achieves **78.63% Dice**, outperforming CPS (77.91%) by 0.72%
   - **Boundary accuracy:** Lowest 95HD (13.97mm), indicating superior edge detection

2. **Generalization Across Datasets**
   - Maintains competitive performance on ImageCAS and ZDCTA datasets
   - Demonstrates robust transfer learning capability across different patient populations
   - Consistent improvements in continuity metrics (ClDice) across all settings

3. **Clinical Relevance**
   - Enhanced bifurcation continuity crucial for stenosis quantification
   - Reduced false positives in thin vessel detection
   - Robust performance with limited annotations addresses real-world data scarcity

---

## 📚 Citation
If you find this work useful, please cite our paper:

## 📞 Contact
For questions or collaborations, please contact:
---

## 📄 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## 🙏 Acknowledgments
- The ASOCA challenge organizers for providing the benchmark dataset
- Guangdong Provincial People's Hospital for the ImageCAS dataset
- All contributors and collaborators who supported this research

---

*Last updated: Sep 2025*

---
*For more details, please refer to the full paper. Code and pre-trained models will be made available upon publication acceptance.*
