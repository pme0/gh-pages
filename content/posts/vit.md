+++

author = "pme0"
title = "Vision Transformer (ViT)"
date = "2022-02-10"
description = ""
tags = [
    "Image",
    "ViT", 
]
math = true

+++



The [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) applies Multi-Headed Self-Attention to sequences of image patches to perform image classification tasks. It is a significant milestone in Machine Learning showing that, with only a few tweaks, a modern architecture is able to perform well in both vision and language domains, thus contributing to unifying Computer Vision and Natural Language Processing at the level of the neural network architecture.

The architecture is depicted below and its components are discussed in turn.

{{< figure src="/images/object-recognition/diagram_ViT.png" width="80%" caption="Fig: The Vision Transformer (ViT) architecture" >}}

##### Patching and Embedding

Applying Transformers to large images is computationally very challenging because the complexity of the Attention mechanism is quadratic in the length of the sequence. Instead of computing attention between image *pixel* sequences, the ViT computes attention between image *patch* sequences, effectively reducing the complexity from $\mathcal{O}(P^2)$ to $\mathcal{O}(p^2)$ where $P$ is the number of pixels and $p \ll P$ is the number of patches.
To make things simple, patches have a fixed size (usually square) and the image width/height should be divisible by the patch size to avoid padding. For example, for an image of size $224 \times 224$ can be represented by a sequence of $196$ patches of size $16 \times 16$.
This means that the sequence to which attention is applied has length $p=196$ (number of patches) as opposed to $P=50,176=224^2$ (number of pixels).

Each image patch is then embedded using a linear projection. Additionally, a learnable position embedding is added to each patch (to retain spatial information) and a learnable class embedding is concatenated with the embedded patch sequence (for classification).

**Tensor dimensions.** The original tensor has shape $[B, C, H, W]$ where $B$ is the batch size, $C$ is the number of channels, $H$ is the image height, $W$ is the image width.

*Patching* reshapes the original tensor to create a patched tensor of shape $[B, H' \times W', C \times p_{H} \times p_{W}]$ where $H'$ and $W'$ are the number of patches in the height and width dimension, respectively; $p_{H}$ and $p_{W}$ are the patche size (in pixels) in height and width dimensions, respectively; the second dimension has been flattened to a 1D array of patches from of a 2D array (grid of patches) and the third dimension has been flattened to a 1D array of patch pixels from a 3D array (2D grid of patch pixels per channel).

*Embedding* linearly projects each of the $H' \times W'$ patches of shape $[C \times p_{H} \times p_{W}]$ to an embedding space of size $D$ and therefore creates a tensor of shape $[B, H' \times W', D]$. To this tensor is added a learnable position embedding of size $D$ for each patch (which does not alter the tensor shape). And finally, the tensor is concatenated with a learnable class embedding of size $D$, resulting in a tensor of size $[B, 1 + H' \times W', D]$.

The table below summarizes the patching and embedding process with a concrete example for a color image of size $224 \times 224$ ($C=3$, $H = W = 224$) split into patches of size $16 \times 16$ ($p_{H} = p_{W} = 16$), and embedding size $512$ ($D=512$). This gives $H' = W' = 14$.

|  operation  |  output size  | example |
|:--:|:--:|:--:|
| input | $[B \times C \times H \times W]$ |  $[1, 3, 224, 224]$
| patching | $[B, H' \times W', C \times p_{H} \times p_{W}]$ |  $[1, 196, 768]$
| patch embedding | $[B, H' \times W', D]$ |  $[1, 196, 512]$
| add position embedding | $[B, H' \times W', D]$ |  $[1, 196, 512]$
| add class token embedding | $[B, 1+ H' \times W', D]$ |  $[1, 197, 512]$


Patchifying an image can be done as follows:
```python
def patchify(x, ps, flatten):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // ps, ps, W // ps, ps) # [B, C, H', pH, W', pW]
    x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H', W', pH, pW, C]
    x = x.flatten(1, 2)  # [B, H' * W', pH, pW, C]
    if flatten_pixels:
        x = x.flatten(2, 4)  # [B, H' * W', C * pH * pW]
    return x
```


Below we show an example of patching for a 
{{< figure src="/images/vit/pexels-pixabay-276517-display.png" width="60%" >}}

##### Transformer

**Attention.** The ViT uses [Multi-Headed Self-Attention (MHSA)](https://arxiv.org/abs/1706.03762)


##### Classifier

...
