# FADConv
FADConv: A Frequency-Aware Dynamic Convolution for Cropland Non-agriculturalization Identification and Segmentation
#
Abstract

Cropland non-agriculturalization refers to the conversion of arable land into non-agricultural uses such as forests, residential areas, and construction sites. This phenomenon not only directly leads to the loss of cropland resources but also poses systemic threats to food security and agricultural sustainability. Accurate identification of cropland and non-cropland areas is crucial for detecting and addressing this issue. Although remote sensing and deep learning methods have shown promise in cropland segmentation, challenges persist in misidentification and omission errors, particularly with high-resolution remote sensing imagery.Traditional CNNs employ static convolution layers, while dynamic convolution studies demonstrate that adaptively weighting multiple convolutional kernels through attention mechanisms can enhance accuracy. However, existing dynamic convolution methods relying on Global Average Pooling (GAP) for attention weight allocation suffer from information loss, limiting segmentation precision. This paper proposes Frequency-Aware Dynamic Convolution (FADConv) and a Frequency Attention (FAT) module to address these limitations. Building upon the foundational structure of dynamic convolution, we designed FADConv by integrating 2D Discrete Cosine Transform (2D DCT) to capture frequency domain features and fuse them. FAT module generates high-quality attention weights that replace the traditional GAP method,making the combination between dynamic convolution kernels more reasonable.Experiments on the GID and Hi-CNA datasets demonstrate that FADConv significantly improves segmentation accuracy with minimal computational overhead. For instance, ResNet18 with FADConv achieves 1.9% and 2.7% increases in F1-score and IoU for cropland segmentation on GID, with only 58.87M additional MAdds. Compared to other dynamic convolution approaches, FADConv exhibits superior performance in cropland segmentation tasks.

Replace the static convolution kernel with FADConv.

For example：

FADConv(in_channels, out_channels, 3, stride=stride, padding=1, num_experts=4),

FADConv(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, num_experts=num_experts)

## 📚 Citation

If you find this repository useful in your research, please consider citing our paper. 🌟

📄 **Paper Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1569843225006612) | 🔗 **DOI:** [10.1016/j.jag.2025.105014](https://doi.org/10.1016/j.jag.2025.105014)

**BibTeX:**

```bibtex
@article{SHU2026105014,
  title     = {FADConv: A frequency-aware dynamic convolution for cropland non-agriculturalization identification and segmentation},
  journal   = {International Journal of Applied Earth Observation and Geoinformation},
  volume    = {146},
  pages     = {105014},
  year      = {2026},
  issn      = {1569-8432},
  doi       = {10.1016/j.jag.2025.105014},
  url       = {https://www.sciencedirect.com/science/article/pii/S1569843225006612},
  author    = {Tan Shu and Li Shen and Yong Wang and Peng Zhang},
  keywords  = {Remote sensing, Cropland non-agriculturalization, Cropland segmentation, Dynamic convolution, Frequency attention, High-resolution images},
}
