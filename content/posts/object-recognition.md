+++

author = "pme0"
title = "Object Recognition"
date = "2022-02-01"
description = ""
tags = [
    "Image",
    "Classification",
    "ViT", 
    "MLPMixer"
]
math = true

+++

{{< figure src="/images/object-recognition/cifar10_grid.png" width="25%" >}}


## Vision Transformer

The [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) applies Multi-Headed Attention to sequences of image patches to perform image classification tasks. It is a significant milestone in Machine Learning, showing that, with only a few tweaks, a modern architecture is able to perform classification tasks in both vision and language domains, thus contributing to unifying Computer Vision and Natural Language Processing at the level of the network architecture used.

The architecture is depicted below and it components are discussed in turn.

{{< figure src="/images/object-recognition/diagram_ViT.png" width="80%" caption="Fig: The Vision Transformer (ViT) architecture" >}}

### Patching and Embedding

Applying Transformers to large images is computationally very challenging because the complexity of the Attention mechanism is quadratic in the length of the sequence. Instead of computing attention between image *pixel* sequences, the ViT computes attention between image *patch* sequences, effectively reducing the complexity from $\mathcal{O}(P^2)$ to $\mathcal{O}(p^2)$ where $P$ is the number of pixels and $p \ll P$ is the number of patches. To make things simple, patches have a fixed size (usually square) and the image width/height should be divisible by the patch size to avoid padding. For example, for an image of size $32 \times 32$ can be represented by a sequence of $64$ patches of size $4 \times 4$. 

Each image patch is then embedded using a linear projection. Additionally, a position embedding is added to each patch (to retain spatial information) and a learnable class embedding is concatenated with the embedded patch sequence (for classification).

### Multi-Headed Attention

### Transformer

### Classifier





## MLP-Mixer