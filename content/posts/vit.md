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

# Introduction

The [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) applies Multi-Headed Self-Attention to sequences of image patches to perform image classification tasks. It is a significant milestone in Machine Learning showing that, with only a few tweaks, a modern architecture is able to perform well in both vision and language domains, thus contributing to unifying Computer Vision and Natural Language Processing at the level of the neural network architecture.

The architecture is depicted below and its components are discussed in turn.

{{< figure src="/images/vit/diagram_ViT.png" width="80%" >}}


Applying Transformers to large images is computationally very challenging because the complexity of the Attention mechanism is quadratic in the length of the sequence. Instead of computing attention between image *pixel* sequences, the ViT computes attention between image *patch* sequences, effectively reducing the complexity from $\mathcal{O}(Q^2)$ to $\mathcal{O}(P^2)$ where $Q$ is the number of pixels and $P \ll Q$ is the number of patches.
To make things simple, patches have a fixed size (usually square) and the image width/height should be divisible by the patch size to avoid padding. For example, for an image of size $224 \times 224$ can be represented by a sequence of $196$ patches of size $16 \times 16$.
This means that the sequence to which attention is applied has length $P=196$ (number of patches) as opposed to $Q=224^2=50,176$ (number of pixels).


# Patching

The input tensor $\bm{X}$ has shape $[B, C, H, W]$ where $B$ is the batch size, $C$ is the number of channels, $H$ is the image height, $W$ is the image width.

*Patching* reshapes each image tensor 
$$
\bm{x} \in \mathbb{R}^{ C \times H \times W}
$$ to create a patched tensor 
$$
\bm{x}\_{p} \in \mathbb{R}^{ N \times C P^2 }
$$
where $H^{\prime}$ and $W^{\prime}$ are the number of patches in the height and width dimension; $P$ is the patch size (in pixels) in both height and width dimensions, i.e. square patches; and $N = H^{\prime} W^{\prime} = H W / P^2$ is the total number of patches.
The first dimension has been flattened to a 1D array of patches from of a 2D array (grid of patches) and the second dimension has been flattened to a 1D array of patch pixels from a 3D array (2D grid of patch pixels per channel).

The patching process can be achieved by reshaping the input tensor `x` as follows:
```python
class Patchify(nn.Module):
    """Convert image into image patches"""
    def __init__(self, patch_size, flatten_pixels=True):
        super().__init__()
        self.patch_size = patch_size
        self.flatten_pixels = flatten_pixels
        
    def forward(self):
        P = self.patch_size
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // P, P, W // P, P) # [B, C, H', P, W', P]
        x = x.permute(0, 2, 4, 3, 5, 1)           # [B, H', W', P, P, C]
        x = x.flatten(1, 2)                       # [B, H' * W', P, P, C]
        if self.flatten_pixels:
            x = x.flatten(2, 4)                   # [B, H' * W', C * P * P]
        return x
```


# Embedding

After being patchified, each image patch is then embedded using a linear projection. Additionally, a learnable position embedding is added to each patch (to retain spatial information) and a learnable class embedding is concatenated with the embedded patch sequence (for classification).

*Embedding* linearly projects each patch tensor of shape $[N, C  P^2]$ to an embedding space of size $D$ and therefore creates an embeded tensor of shape $[N, D]$. 
This tensor is then concatenated with a learnable class embedding of size $D$, resulting in a tensor of size $[1 + N, D]$.
And finally, is added a learnable position embedding, also of size $D$, which does not alter the tensor shape.  
The embedding process can be defined as
$$
\bm{z}\_{0} = [\bm{x}\_{\text{class}}, x_{p}^{1} \bm{E}, \dots, x_{p}^{N} \bm{E}] + \bm{E}_{\text{pos}}
$$
where 
$\bm{z}\_{0} \in \mathbb{R}^{(1 + N)\times D}$ is the embedded sequence of patches from a single image;
$\bm{x}\_{\text{class}} \in \mathbb{R}^{D}$ is the class embedding; 
$\bm{x}\_{p}^{k} \in \mathbb{R}^{ C P^2 }$ is the $k$th patch; 
$\bm{E} \in \mathbb{R}^{C P^2 \times D}$ is the linear projection tensor used to embed the patches; and
$\bm{E}\_{\text{pos}} \in \mathbb{R}^{(1 + N)\times D}$ is the positional embedding.


```python
class LinearEmbedding(nn.Module):
    """Embed patch using a linear layer"""
    def __init__(self, num_channels, patch_size, embed_dim):
        super().__init__()
        self.linear_embedding = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

    def forward(self, x, batch_size):
        return self.linear_embedding(x)
    

class ClassEmbedding(nn.Module):
    """Append class embedding"""
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x batch_size):
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        return x
    

class PositionalEmbedding(nn.Module):
    """Add positional embedding"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        
    def forward(self, x, num_patches):
        x += self.pos_embedding[:, : num_patches + 1]
        return x
```

Gathering the three embedding stepss in a module:
```python
class Embedding(nn.Module):
    """Embed image patches"""
    def __init__(self, num_channels, image_size, patch_size, embed_dim):
        super().__init__()
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.linear_embedding = LinearEmbedding(num_channels, patch_size, embed_dim).linear_embedding
        self.class_embedding = ClassEmbedding(embed_dim)
        self.positional_embedding = PositionalEmbedding(self.num_patches, embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear_embedding(x)
        x = self.class_embedding(x, batch_size)
        x = self.positional_embedding(x, self.num_patches)
        return x
```


# An example of patching & embedding

The patching process, shoqn below, splits an image into patches and the embedding process then takes each patch separately and embeds it into a 1D embedding space.

{{< figure src="/images/vit/pexels-pixabay-276517-patches.png" width="60%" >}}


Below we give a concrete example for a color image of size $H \times W = 224 \times 224$ with $C=3$ channels, split into square patches of size $P = 16$, and embedding size $D=512$. This gives the same number of patches in each dimension $H^{\prime} = W^{\prime} = 14$, for a total number of patches $N = H^{\prime} \times W^{\prime} = 196$.
The table summarizes the tensor dimensions throughout the patching and embedding processes:

|  layer  |  input shape  |  output shape  |  example output  |
|:--:|:--:|:--:|:--:|
| input | - | $[B \times C \times H \times W]$ | $[1, 3, 224, 224]$ |
| patchify | $[B \times C \times H \times W]$ | $[B, N, C P^2 ]$ | $[1, 196, 768]$ |
| linearly embedding | $[B, N, C P^2 ]$ |  $[B, N, D]$ | $[1, 196, 512]$ |
| append class token embedding | $[B, N, D]$ | $[B, 1+N, D]$ |  $[1, 197, 512]$ |
| add positional embedding | $[B, 1+N, D]$ | $[B, 1+N, D]$ |  $[1, 197, 512]$ |



and we can confirm the dimensions using PyTorch *Forward Hooks*:
```python
# define parameters
image_size = (224,224)   # W, H
num_channels = 3         # C
patch_size = 16          # P
embed_dim = 512          # D
batch_size = 1           # B

# initiate patching and embedding modules
patchify = Patchify(patch_size)
embed = Embedding(num_channels, image_size, patch_size, embed_dim)

# define hook function
shapes = {}
def getShape(name):
    def hook(model, input, output):
        shapes[name] = (tuple(input[0].detach().shape), tuple(output.detach().shape))
    return hook

# register hooks
h1 = patchify.register_forward_hook(getShape('patchify'))
h2 = embed.linear_embedding.register_forward_hook(getShape('linear_embedding'))
h3 = embed.class_embedding.register_forward_hook(getShape('class_embedding'))
h4 = embed.positional_embedding.register_forward_hook(getShape('positional_embedding'))

# run forward pass to collect data from hooks
x = torch.ones((batch_size, num_channels, image_size[0], image_size[1]))
shapes["input"] = (tuple(x.shape), tuple(x.shape))
y = embed(patchify(x))
```
The dictionary `shapes` contains the shape of input and output tensors to each hooked layer:
```python
{
    'patchify': ((1, 3, 224, 224), (1, 196, 768)),
    'linear_embedding': ((1, 196, 768), (1, 196, 512)),
    'class_embedding': ((1, 196, 512), (1, 197, 512)),
    'positional_embedding': ((1, 197, 512), (1, 197, 512))
}
```


# Transformer

The ViT uses [Multi-Headed Self-Attention (MHSA)](https://arxiv.org/abs/1706.03762). 
The self-attention mechanism is the central component of the transformer. It models the interactions between entities in a sequence as described below.
The goal of self-attention is to capture the interaction amongst $n$ entities by encoding each entity in terms of the global contextual information.

Let $\bm{X} \in \mathbb{R}^{N \times D}$ denote a sequence of $N$ tokens $(\bm{x}\_{1},\dots,\bm{x}\_{N})$, where $D$ is the dimension of the embedding/token used to represent each entity.
A *token* here is an embedding of an image patch. The batch size $B$ is omitted for simplicity, and so $N$ denotes the sequence length of one sample (number of patches per image), not the number of sequences (number of images).

**Self-Attention.**
To capture the interaction amongst the $N$ entities we define three learnable weight matrices to transform entities into 
*Queries*, *Keys* and *Values* (respectively $\bm{W}\_{Q}, \bm{W}\_{K}, \bm{W}\_{V} \in \mathbb{R}^{D \times d}$). 
The input sequence $\bm{X}$ is first projected onto these weight matrices to obtain 
$$\begin{align*}
    \bm{Q} &= \bm{X} \bm{W}\_{Q} \\\
    \bm{K} &= \bm{X} \bm{W}\_{K} \\\
    \bm{V} &= \bm{X} \bm{W}\_{V} \\\
\end{align*}$$
with 
$\bm{Q}, \bm{K}, \bm{V} \in \mathbb{R}^{N \times d}$.
Usually $d < D$, that is, the dimensions of the representation is smaller than the dimension of the original embedding.
The output of the self attention layer is then given by
$$\bm{Z} 
    = \text{Attention}(\bm{Q},\bm{K},\bm{V})
    = \text{softmax}\left( \frac{\bm{Q} \bm{K}^{T}}{\sqrt{d}} \right) \bm{V}
$$
with $\bm{Z} \in \mathbb{R}^{N \times d}$.
The *softmax* term corresponds to the *attention scores* (scaled and normalised): if the dot product $\bm{Q} \bm{K}^{T}$ is large for two given entities, they co-occur often and their attention score is high.


**Multi-Headed Self-Attention.**
The original authors of the [self-attention mechanism](https://arxiv.org/abs/1706.03762) projected the queries, keys and values $h$ times with *different*, learnable linear projections. The embedding is split evenly between the $h$ heads so that $d=D/h$ for each head.
With multiple attention heads, each head learns a different representation of the same input. 

$$\begin{align*}
&\bm{Z}\_{i} = \text{Attention}(\bm{Q}\_{i}, \bm{K}\_{i}, \bm{V}\_{i})\\\
&\bar{\bm{Z}} = \text{concat}(\bm{Z}\_{1},\dots, \bm{Z}\_{h}) \\\
&\bm{Z} = \text{MultiHeadAttention}(\bm{Q},\bm{K},\bm{V}) = \bar{\bm{Z}} \bar{\bm{W}} \\\ 
\end{align*}$$
with matrices
$\bm{Q}\_{i}, \bm{K}\_{i}, \bm{V}\_{i} \in  \mathbb{R}^{D \times d}$; 
head scores $\bm{Z}\_{i} \in  \mathbb{R}^{N \times d}$;
concatenated scores $\bar{\bm{Z}} \in  \mathbb{R}^{N \times D}$;
output weight matrix $\bar{\bm{W}} \in  \mathbb{R}^{D \times D}$; 
output representation $\bm{Z} \in  \mathbb{R}^{N \times D}$; 
and $h$ is the number of attention heads ($d=D/h$).
The MHSA mechanism is depicted schematically below:

{{< figure src="/images/vit/diagram_MHSA.png" width="60%" >}}



**Linear Layer.** ...

**Normalization Layer.** ...


The transformer can then be defined as 
$$\begin{align*}
\bm{z}\_{l}^{\prime} &= \text{MHSA}(\text{LN}(\bm{z}\_{l−1})) + \bm{z}\_{l−1} \\\
\bm{z}\_{l} &= \text{MLP}\_{\text{att}}(\text{LN}(\bm{z}\_{l}^{\prime})) + \bm{z}\_{l}^{\prime} \\\
\end{align*}$$
where $l=\\{1,\dots,L\\}$ is the layer index; ....


# Classifier

The transformer forward pass can then be defined as 
$$
\bm{y} = \text{MLP}\_{\text{cls}}(\text{LN}(\bm{z}\_{L}^{0}))
$$
where $\bm{y}$ ........ ; $\bm{z}\_{L}^{0}$ is the image representation to each the classification layer is attached; $\bm{z}\_{0}^{0} = \bm{x}\_{\text{class}}$; 



# Trainable Parameters

The number of parameters in the patching and embedding process is:
```python
def count_params(layers):
    total_params = 0
    for layer in layers:
        params = sum(p.numel() for p in eval(layer).parameters() if p.requires_grad)
        total_params += params
        print("{}: {:,}".format(eval(layer).__class__.__name__, params))
    print("Total: {:,}".format(total_params))
        
layers = ["patchify", "embed.linear_embedding", 
    "embed.class_embedding", "embed.positional_embedding"]
count_params(layers)
```

gives a total of 495,104 parameters. The Linear Embedding layer is responsible for most of those:

{{< highlight python "hl_lines=5" >}}
Patchify: 0
Linear: 393,728
ClassEmbedding: 512
PositionalEmbedding: 100,864
Total: 495,104
{{< / highlight >}}

