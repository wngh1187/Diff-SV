# Diff-SV

Pytorch code for following paper:

* **Title** : Diff-SV: A Unified Hierarchical Framework for Noise-Robust Speaker Verification Using Score-Based Diffusion Probabilistic Models (submitted to ICASSP 2024)
* **Autor** : Ju-ho Kim, Jungwoo Heo, Hyun-seo Shin, Chan-yeong Lim and Ha-Jin Yu


# Abstract
<img align="middle" width="2000" src="https://github.com/wngh1187/Diff-SV/blob/main/fig1_6.png">
Background noise considerably reduces the accuracy and reliability of speaker verification (SV) systems. 
These challenges can be addressed using a speech enhancement system as a front-end module. 
Recently, diffusion probabilistic models (DPMs) have exhibited remarkable noise-compensation capabilities in the speech enhancement domain. 
Building on this success, we propose Diff-SV, a noise-robust SV framework that leverages DPM. 
Diff-SV unifies a DPM-based speech enhancement system with a speaker embedding extractor, and yields a discriminative and noise-tolerable speaker representation through a hierarchical structure. 
The proposed model was evaluated under both in-domain and out-of-domain noisy conditions using the VoxCeleb1 test set, an external noise source, and the VOiCES corpus. 
The obtained experimental results demonstrate that Diff-SV achieves state-of-the-art performance, outperforming recently proposed noise-robust SV systems. 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:21.04-py3' image of Nvidia GPU Cloud for conducting our experiments. 
* Run shell file to make docker image
```
./docker/build.sh
```
