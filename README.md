# DA6401 Visual Perception Pipeline: Oxford-IIIT Pet

This repository contains a complete multi-task visual perception pipeline, implementing classification, object localization, and semantic segmentation using a shared VGG11 backbone.

**W and B Report Link:** [DA6401_MOHANRAJ_Assignment2](https://api.wandb.ai/links/ed21d022-indian-institute-of-technology-madras/8tqkn0g5)

---

### Project Analysis & Key Findings

#### 2.1 Batch Normalization (BN)
* **Stability:** Forced activations into stable bands, eliminating internal covariate shift.
* **Speed:** Accelerated convergence by smoothing the loss landscape.
* **Learning Rate:** Enabled higher stable learning rates (up to 0.001) without divergence.
* **Gradient Health:** Prevented vanishing/exploding gradients in deep ReLU layers.

#### 2.2 Dropout & Generalization
* **Overfitting:** Models without dropout ($p=0$) showed rapid training but high validation volatility.
* **Redundancy:** $p=0.2$ forced the network to learn robust, non-reliant features.
* **Consistency:** $p=0.5$ achieved the most stable validation loss over long durations.
* **Gap Control:** Higher dropout effectively narrowed the gap between training and validation error.

#### 2.3 Transfer Learning
* **Frozen Backbone:** Lowest performance; ImageNet filters lack pixel-precise spatial awareness.
* **Partial Fine-Tuning:** Massive boost by unfreezing Block 5 to adapt to pet anatomy.
* **Full Fine-Tuning:** Best Dice scores; allowed end-to-end alignment for dense segmentation.
* **Specificity:** Confirmed that deep layers must be unfrozen to transition from edges to shapes.

#### 2.4 Feature Map Visualization
* **Low-Level:** Layer 1 maps capture edges, whiskers, and crisp geometry.
* **High-Level:** Layer 5 maps show abstract semantic heatmaps (ears, snouts, eyes).
* **Hierarchy:** Demonstrated a clear transition from localized lines to global pet structures.
* **Filtering:** Deeper layers successfully ignore background noise to focus on subject identity.
* **Code link:** [2.4 code snippet] (https://github.com/MohanRaj479/DA6401_MOHANRAJ_Assignment2/blob/main/2.4_2.6_2.7_Base_codes.py) 
#### 2.5 Object Detection (IoU)
* **Precision:** Achieved high IoU by integrating coordinate regression with VGG11 features.
* **Failure Case:** Low-contrast subjects (black pets on dark backgrounds) caused box shifting.
* **Confidence vs. IoU:** High confidence doesn't guarantee IoU if spatial edges are ambiguous.
* **Regression:** Spatial coordinate accuracy relies heavily on clear boundary detection.

#### 2.6 Dice vs. Pixel Accuracy
* **Accuracy Bias:** Pixel accuracy was misleadingly high ($>75\%$) due to background dominance.
* **Dice Utility:** Ignored True Negatives to focus purely on foreground overlap accuracy.
* **Imbalance:** Proved that Dice is the superior metric for tasks with rare "border" classes.
* **Optimization:** Dice forces the model to actually find the pet rather than "lazy-predicting" background.
* **Code link:** [2.6 code snippet] (https://github.com/MohanRaj479/DA6401_MOHANRAJ_Assignment2/blob/main/2.4_2.6_2.7_Base_codes.py) 
#### 2.7 Pipeline Generalization
* **In-the-Wild:** Robust localization on internet images despite non-standard poses.
* **Boundary Sensitivity:** Segmentation masks occasionally bled into complex, cluttered backgrounds.
* **Semantic Transfer:** High-level breed understanding transferred well to novel data.
* **Environment:** Pixel-level precision is still sensitive to lighting outside the training distribution.
* **Code link:** [2.7 code snippet] (https://github.com/MohanRaj479/DA6401_MOHANRAJ_Assignment2/blob/main/2.4_2.6_2.7_Base_codes.py) 
#### 2.8 Meta-Analysis & Reflection
* **Critical BN:** BatchNorm was the single most important factor for multi-task stability.
* **Regularization:** $p=0.5$ Dropout was essential to control the 4096-dim classifier head.
* **Task Synergy:** Full fine-tuning resolved conflicts between classification and spatial tasks.
* **Loss Strategy:** Weighted Cross-Entropy was required to overcome 75% background imbalance.
