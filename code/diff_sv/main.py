import os
import random
import importlib
import numpy as np
import torch
import torch.nn as nn

import arguments
import trainers.train as train
import data.data_loader as data
from data.voxceleb1 import VoxCeleb1
from data.voices import Voices
from optimizer.CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
from models.diffsv import DiffSV
from log.controller import LogModuleController
from data.preprocessing import DataPreprocessor

def set_experiment_environment(args):
    #### reproducible ###
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = args['flag_reproduciable']
    torch.backends.cudnn.benchmark = not args['flag_reproduciable']

    ### DDP env ###
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4021'
    args['rank'] = args['process_id']
    args['device'] = 'cuda:' + args['gpu_ids'][args['process_id']]
    torch.cuda.empty_cache()
    torch.cuda.set_device(args['device'])
    if len(args['gpu_ids']) >1:
        torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])

def run(process_id, args):
    ### check parent process ###
    args['process_id'] = process_id
    args['flag_parent'] = process_id == 0
    
    ### experiment environment ###
    set_experiment_environment(args)
    trainer = train.ModelTrainer(args)
    trainer.args = args
    
    ### set logger ###
    if args['flag_parent']:
        logger = LogModuleController.Builder(args['name'], args['project']
        ).tags(args['tags']
        ).description(args['description']
        ).save_source_files(args['path_scripts']
        ).use_local(args['path_logging']
        #).use_wandb(args['wandb_group'], args['wandb_entity'] #Activate if using WandB (https://wandb.ai)
        ).build()
        trainer.logger = logger
    
    ### set dataset ###
    trainer.vox1 = VoxCeleb1(
        args['path_vox1_train'], 
        args['path_vox1_test'], 
        f'{args["path_vox1_test"]}_noise', 
        args['path_vox1_trials']
    )
    args['num_speaker'] = 1211

    trainer.voices = Voices(
        args['path_voices_dev'],
        args['path_voices_dev_trials'],
        args['path_voices_eval'],
        args['path_voices_eval_trials']
    )

    ### set data loader ###
    loaders = data.get_loaders(args, trainer.vox1, trainer.voices)
    trainer.train_set_sampler = loaders[0]
    trainer.train_loader = loaders[1]
    trainer.enrollment_set = loaders[2]
    trainer.enrollment_loader = loaders[3]
    trainer.dev_loader_voices = loaders[4]
    trainer.eval_loader_voices = loaders[5]

    ### set model ###
    model = DiffSV(args).to(args['device'])

    if args['flag_parent']:
        args['model_num_param'] = model.nparams
        print('num of model params: ', args['model_num_param'])
        trainer.logger.log_parameter(args)

    ### set criterion ###
    criterion = {}
    classification_loss_function = importlib.import_module('loss.'+ args['classification_loss']).__getattribute__('LossFunction')
    criterion['classification_loss'] = classification_loss_function(args['code_dim'], args['num_speaker']).to(args['device']) 
    criterion_params = list(criterion['classification_loss'].parameters())

    ### set ddp if using multi gpu ###
    if len(args['gpu_ids']) >1:   
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args['device']], find_unused_parameters=True)
        criterion['classification_loss'] = nn.parallel.DistributedDataParallel(criterion['classification_loss'], device_ids=[args['device']], find_unused_parameters=True)

    trainer.model = model
    trainer.criterion = criterion

    ### set optimizer ###
    trainer.optimizer = torch.optim.Adam(
        list(model.parameters()) + criterion_params, 
        lr=args['lr_start'], 
        weight_decay=args['weigth_decay'],
        amsgrad = args['amsgrad']
    )

    ### set lr scheduler ###
    args['number_iteration'] = len(trainer.train_loader)
    trainer.lr_scheduler = CosineAnnealingWarmUpRestarts(
        trainer.optimizer, 
        T_0=args['number_iteration'] * args['number_cycle'], 
        T_mult=args['T_mult'], 
        eta_max=args['eta_max'],  
        T_up=args['number_iteration'] * args['warmup_number_cycle'], 
        gamma=args['gamma'])

    ### train and test model ###
    trainer.run()
        
    if args['flag_parent']:    trainer.logger.finish()


if __name__ == '__main__':
    ### get arguments ###
    args = arguments.get_args()

    ### check dataset ###
    data_preprocessor = DataPreprocessor(args['path_musan'], args['path_nonspeech'], args['path_vox1_test'])
    data_preprocessor.check_environment()

    ### set gpu device ###
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        args['gpu_ids'] = args['usable_gpu'].split(',')
    
    if len(args['gpu_ids']) == 0 or len(args['gpu_ids']) == 1:
        run(0, args)
    else:
        ### set DDP ###
        args['world_size'] = len(args['gpu_ids'])
        args['batch_size'] = args['batch_size'] // (args['world_size'])
        args['num_workers'] = args['num_workers'] // args['world_size']
        
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.spawn(
            run, 
            nprocs=args['world_size'], 
            args=(args,)
        )