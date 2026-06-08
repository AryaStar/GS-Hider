# Introduction

GS-Hider with SE-Net Enhanced Decoder. Built upon [GS-Hider](https://github.com/xuanyuzhang21/GS-Hider), this project improves the decoder using SE-Net to decouple original and steganographic scene features.

# Install with feature rasterizer

```
pip install submodule/diff-gaussian-rasterization
pip install submodule/simple-knn
```

# Train
During training, the camera poses of training images for the original and steganographic scenes must be strictly matched. This can be achieved by rendering both 3DGS models with identical camera poses.
```
bash script.sh
```