import os
from itertools import chain
import torch

def get_args():
    system_args = {
        # expeirment info
        'project'       : 'NoiseRobustSV',
        'name'          : 'diff-sv',
        'tags'          : ['DFSV'],
        'description'   : 'train and test diff-sv',

        # local
        'path_logging'  : '/results',

        # wandb
        'wandb_group'         : None, 
        'wandb_entity'        : None, 

        # VoxCeleb1 DB
        'path_vox1_train'   : '/data/VoxCeleb1/train',
        'path_vox1_test'    : '/data/VoxCeleb1/test',
        'path_vox1_trials'  : '/data/VoxCeleb1/trials.txt',

        # noise DB
        'path_musan'        : '/data/musan',
        'path_nonspeech'    : '/data/Nonspeech',
        'path_rir'          : '/data/rir_noises/simulated_rirs/',

        # VOiCES DB
        'path_voices_dev'   : '/data/VOiCES_Box_unzip/Development_Data/Speaker_Recognition/sid_dev',
        'path_voices_dev_trials': '/data/VOiCES_Box_unzip/Development_Data/Speaker_Recognition/sid_dev_lists_and_keys/dev-trial-keys.lst',
        'path_voices_eval': '/data/VOiCES_Box_unzip/Speaker_Recognition/sid_eval/',
        'path_voices_eval_trials': '/data/VOiCES_Box_unzip/VOiCES_challenge_2019_post-eval-release/VOiCES_challenge_2019_eval.SID.trial-keys.lst',

        # device
        'num_workers'   : 12,
        'usable_gpu'    : '0,1,2,3',
        'tqdm_ncols'    : 90,
        'path_scripts'     : os.path.dirname(os.path.realpath(__file__))
    }
    
    experiment_args = {
        # env
        'epoch'                     : 320,
        'batch_size'                : 160,
        'number_iteration_for_log'  : 20,
        'rand_seed'                 : 777,
        'flag_reproduciable'        : False,

        # optimizer
        'amsgrad'                   : True,
        'lr_start'                  : 1e-7,
        'lr_end'                    : 1e-7,
        'number_cycle'              : 40,
        'warmup_number_cycle'       : 1,
        'T_mult'                    : 1.5,
        'eta_max'                   : 1e-2,
        'gamma'                     : 0.5,
        'weigth_decay'              : 1e-4,

        # criterion
        'classification_loss'                               : 'aam_softmax',

        # model
        'first_kernel_size'     : 7,
        'first_stride_size'     : (2,1),
        'first_padding_size'    : 3,
        'l_channel'             : [16, 32, 64, 128],
        'l_num_convblocks'      : [3, 4, 6, 3],
        'code_dim'              : 128,
        'stride'                : [1,2,2,1],

        # data
        'winlen'            : 400,
        'winstep'           : 160,
        'train_frame'       : 198,
        'nfft'              : 512,
        'samplerate'        : 16000,
        'nfilts'            : 80,
        'premphasis'        : 0.97,
        'f_min'             : 20,
        'f_max'             : 7600,
        'winfunc'           : torch.hamming_window,
        'DA_p'              : 1,
        'test_frame'        : 998,
        'num_seg'           : 4
    }

    # set args (system_args + experiment_args)
    args = {}
    for k, v in chain(system_args.items(), experiment_args.items()):
        args[k] = v

    return args