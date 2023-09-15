# Diff-SV

Pytorch code for following paper:

* **Title** : Diff-SV: A Unified Hierarchical Framework for Noise-Robust Speaker Verification Using Score-Based Diffusion Probabilistic Models (submitted to ICASSP 2024)
* **Autor** : Ju-ho Kim, Jungwoo Heo, Hyun-seo Shin, Chan-yeong Lim and Ha-Jin Yu


# Abstract
<img align="middle" width="1800" src="https://github.com/wngh1187/Diff-SV/blob/main/fig1_6.png">
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
* Run 'build.sh' file to make docker image
```
./docker/build.sh
```

* Run shell file to activate docker container
* Note that you must modify the mapping path before running the 'interactive.sh' file

```
./docker/interactive.sh
```

## Datasets
* We used [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dataset for training and test. 
* For noisy test, we used the [MUSAN](https://www.openslr.org/17/), [Nonspeech100](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html), and [VOiCES](https://iqtlabs.github.io/voices/downloads/) datasets.
* Each downloaded dataset should be mapped to the 'data' folder in docker environment.

# Train and test
```
python3 code/diff_sv/main.py 
```


# Citation
Please cite this paper if you make use of the code. 
```
TBA
```
