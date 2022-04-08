---
title : NeRF
notetype : feed
date : 08-04-2022
---

- title: "NeRF: Representing scenes as neural radiance fields for view synthesis"
- authors: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
- year: 2020


---
title: "NeRF: Representing scenes as neural radiance fields for view synthesis"
authors: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
year: 2020
aliases: [NeRF]
---
# NeRF: Representing scenes as neural radiance fields for view synthesis

- [ ] What is radiance - https://graphics.fandom.com/wiki/Radiance
- [ ] Overview video - https://youtu.be/CRlN-cYFxTk
- [ ] Talk by author [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://www.youtube.com/watch?v=iKyIJ_EtSkw)
- [ ] https://medium.com/swlh/nerf-neural-radiance-fields-79531da37734
- [ ] A lot of [[📃 NeRF_ Representing scenes as neural radiance fields for view synthesis - mildenhall_nerf_2020|NeRF]] follow ups - [Jon Barron](https://jonbarron.info/)


## Overview
- Category - paper proposes a new method for representing 3D scenes with purpose of the synthesis of novel views.
- Context:
    - novel view synthesis work with neural networks
    - classic volume rendering (2d projecton of a 3D discretely sampled data set)
    - Computer graphics and rendering based on real world imagery.
- Contributions
    - Approach for representing **continuous** scenes with complex geometry and materials as 5D (x, y, z location and $\theta$, $\phi$ observation angle) neural radiance fields, parameterised as basic MLP networks (multilayer perceptron aka fully connected NN).
        - As opposed to discrete representations of previous works.
    - A differentiable rendering procedure based on classic rendering techniques, whish is used to optmize the representations from standard RGB images (with known camera locations). This includes a hierarchical sampling strategy to allocate the MLP's capacity towards space with visible scene content.
    - A positional encoding to map each input 5D coordinate into a higher dimensional space, which enables successfull optimization of neural radiance fields to represent high-frequency scene content.
    - Summarized
        - Method for compact representations of high-fidelity 3D scenes with consistent lighting for novel view syntehsis and other applications.
- Correctness
    - The published implementations of paper seem to confirm its working quality.
    - As paper was released in 2020, there was a large number of follow-up papers, it is highly cited.
- Clarity
    - Paper is very clearly written, and easy to understand.


- References I haven't seen
    - Approaches compared with
        - [ ] Neural Volumes
        - [ ] Scene Representation Networks (better-performing followup to [[Deep voxels]])
        - [ ] Local Light Field Fusion
    - [ ] Lumigraph
    - [ ] Ray Tracing of Volume Data
    - [ ] DeepSDF

## Approach

### What

As input, NeRF takes a sparse set of RGB images (more - better) with known camera locations/parameters. NeRF trains a new NN for each scene, with the trained NN being a continuos representation of the 3D scene. The network is then able to be used to produce novel views, or establish a discrete representation of the 3D scene.

### Network

Network is fully-connected NN, or MLP. No convolutional layers are involved.

As input, network takes 5 values - $x, y, z, \theta, \phi$. $x, y, z$ are coordinates of a point in space. $\theta, \phi$ define a viewing direction.
Viewing direction is represented as a vector in spherical coordinates - $(r, \theta, \phi)$ - the radial distance, azimuthal angle, and polar angle, but without the radial distance.

![[Pasted image 20220317163300.png]]

View dependence is required, as lighting bouncing off objects is visible differently based on the viewing angle, it affects the radiance and the viewed colors.

As output network gives 4 values: $\sigma$ - the volume density at the point, and emited RGB radiance viewed from the given direction.

Since density at a point shouldn't depend on viewing direciton, the density output occurs earlier in the network, with only x,y,z as input. After the density is produced, viewing direciton input is added, and RGB values are produced from all 5 input values:

![[Pasted image 20220317164502.png]]

![[Pasted image 20220317163407.png]]

### Training

Because we are only given images, we don't have information about point volume density or the radiance at those 3D points. Hence, there's no direct labels for the model to train on.

Simplified significantly, here's what ends up happening:
For each pixel of each image, a ray is produced, from the camera to an end of the observation range. Points are sampled along the ray. For each point, network gives a prediction of $\sigma$ and RGB. That can be done because $x, y, z$ are known and $\theta, \phi$ are given by the camera ray for the pixel.
![[Pasted image 20220317165116.png|300]]

For the produced densities and RGB values, volume ray-based rendering can be applied for that specific pixel. We get a prediction of what the pixel looks like based on model predictions. And we have the ground truth of the actual pixel of the image. The difference between that prediction and the g.t. is the loss we are optimising.

Estimating the color of a ray actually involves integration along the ray length, where the volume density is considered to be a probability of ray of light bouncing back of a particle. In this paper, the integral is calculated by quadrature (?) from the point samples. The ray is split into bins and samples are taken from those bins based on uniform distribution. That is done instead of selecting point samples uniformly along the ray. If we took samples uniformly along the ray, we would unnecessarily restrict the resolution of data, even though we actually have information about entire continuous space. Sampling randomly helps to get into more of a smooth representation, with more information extracted then from sparce regular samples.

- [ ] TODO look into actual math #task
[[TODO rename - 3D volume rendering by accumulating values of rays of different opacities]]

The rendering looks something like this:
Accumulate the information of color based on the density at each point.
![[Pasted image 20220317170036.png]]

The process of volume rendering is differentiable, so we can backpropagate the loss to update the model weights.

[[TODO rename - If can put an appropriate bottleneck in neural network and differentiate throughout - should be able to learn the required representation]]. We put a very specific requirement for intermediate values produced by the network to result in good rendering after the rendering process. Even though we don't have information about the 3D space directly, we end up training network to get it from images, based on our image ground truth.

### Novel sampling

Once the network is trained, we just need to perform volume rendering by ray for each pixel using the trained network for volume sampling. Essentially replicating the trianing process, but without the backpropagation.

### Positional encoding

[[Neural networks are biased towards learning low-frequency functions]]. Because of that, if models are trained directly on $x,y,z,\theta,\phi$ they converge to low-frequency functions. That means that they capture general shapes and colors of the scene, but fail to capture fine-grain high-fidelity sharp visual features.

![[Pasted image 20220317170438.png]]

To eleviate that, paper proposes a positional encoding of 5D ($x,y,z,\theta,\phi$) vectors to a higher-dimensional space using high-frequency functions.

Before being passed to the network, values are encoded into value in $\mathbb{R}^{2L}$ by using function $\gamma$ (defined below) and then passed to the network.

$$
\gamma(p) = (sin(2^0\pi p), cos(2^0\pi p), ...,
             sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))
$$
With parameter $L$.

Specifically, the function $\gamma$ is applied separately to each of the three coordinate values in **x** (which are normalised to lie in $[-1, 1]$) and to the three components of the Cartesian representation of the viewing direction unit vector **d** (which by construction lie in $[-1, 1]$).

In writer's experiments, they set $L = 10$ for $\gamma(x)$ and $L = 4$ for $\gamma(d)$.

- The [[📃 Fourier features let networks learn high frequency functions in low dimensional domains - tancik_fourier_2020|Fourier trick paper]] was a followup to [[📃 NeRF_ Representing scenes as neural radiance fields for view synthesis - mildenhall_nerf_2020|NeRF]], because they were confused why the actual NeRF trick works. Turns out that the trick they used is a hard-coded variant of a better general option.

### Hierarchical volume sampling

## Pass 3 - Try implementation

- [ ] Pass 3 - try implementation

- ["Tiny NeRF" Colab nb](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
- [NeRF Keras Implementation + tutorial](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/nerf.ipynb)
- [PyTorch NeRF implementation](https://github.com/yenchenlin/nerf-pytorch)

## NeRF Follow-ups

- NeRF-W - "NeRF in the Wild"
    - [[📃 NeRF in the wild_ Neural radiance fields for unconstrained photo collections - martin-brualla_nerf_2021]]
- D-NeRF
- Nerfy
- NeRF for medical applications
    - [[📃 MedNeRF_ Medical neural radiance fields for reconstructing 3D-aware CT-Projections from a single x-ray - corona-figueroa_mednerf_2022]]
- Block NeRF
    - [[📃 Block-NeRF_ Scalable large scene neural view synthesis - tancik_block-nerf_2022]]
- Nvidia presents "Instant NeRF"
    - https://github.com/NVlabs/instant-ngp - "Instant Neural Geometrical Primitives"
    - Fast training of NeRFs using NVIDIA's cuda with gpus and new technique, designed for it - "Multiresolution Hash Encoding"
    - [[📃 Instant neural graphics primitives with a multiresolution hash encoding - muller_instant_2022]]

## Permanent notes
- [[Computer Graphics]]
    - [ ] [[Rendering simulates physical process of image formation to produce a 2D view of a 3D scene]]
    - [ ] [[TODO rename - 3D volume rendering by accumulating values of rays of different opacities]]
    - [ ] Something about radiance of a certain point depending on the viewing direction
- [ ] What is a neural radiance field
    - Neural - because neural network, MLP
    - Radiance - that's what they are learning to predict
    - Field - bacause continuous
- [ ] [[TODO rename - If can put an appropriate bottleneck in neural network and differentiate throughout - should be able to learn the required representation]]
- [ ] Something on the network somehow understanding space
- [ ] Training network to learn functions
- [ ] Frequencies of functions
- [ ] [[Neural networks are universal function approximators]]
- [ ] [[Neural networks are biased towards learning low-frequency functions]]

## References
- [Volume rendering](https://en.wikipedia.org/wiki/Volume_rendering)