#!/bin/bash
# docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/Diff-SV/ -v /home/juho/hdd1/exp_result:/results -v /home/juho/DB/VoxCelebs:/data dfsv
#docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/Diff-SV/ -v /home/juho/hdd1/exp_result:/results -v /home/juho/DB/VoxCelebs:/data dfsv
docker run --runtime=nvidia --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -v $PWD:/workspace/Diff-SV/ -v /ssd1/exp_result:/results -v /ssd1/DB/UNU_env/dataset:/data diff_sv
