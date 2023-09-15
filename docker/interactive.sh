#!/bin/bash
docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/Diff-SV/ -v 'define your save path':/results -v 'define your data path':/data diff_sv
