import random
import math
import copy
import torch
import torch.utils.data as td
import soundfile as sf
import numpy as np

import utils.util as util
from data.noise_aug import Noises
from data.reverberation import RIRReverberation
from torch_audiomentations import Compose, Gain, AddColoredNoise, PitchShift

def get_loaders(args, vox1, voices):
    train_set = TrainSet(args, vox1.train_set)
    test_set = TestSet(args, vox1.test_set)
    dev_set_voices = TestSet_voices(args, voices.dev_set)
    eval_set_voices = TestSet_voices(args, voices.eval_set)
    if len(args['gpu_ids']) >1:
        train_set_sampler = td.DistributedSampler(train_set, shuffle=True)    
        test_set_sampler = td.DistributedSampler(test_set, shuffle=False)
        dev_set_sampler_voices = td.DistributedSampler(dev_set_voices, shuffle=False)
        eval_set_sampler_voices = td.DistributedSampler(eval_set_voices, shuffle=False)
    else:
        train_set_sampler = None
        test_set_sampler = None
        dev_set_sampler_voices = None
        eval_set_sampler_voices = None
    

    train_loader = td.DataLoader(
        train_set,
        batch_size=args['batch_size'],
        pin_memory=True,
        num_workers=args['num_workers'],
        sampler=train_set_sampler,
        drop_last=True,
        shuffle=False if train_set_sampler is not None else True
    )

    test_loader = td.DataLoader(
        test_set,
        batch_size=args['batch_size']//4,
        pin_memory=True,
        num_workers=args['num_workers']//2,
        sampler=test_set_sampler,
        shuffle = False
    )

    dev_loader_voices = td.DataLoader(
        dev_set_voices,
        batch_size=args['batch_size']//4,
        pin_memory=True,
        num_workers=args['num_workers']//2,
        sampler=dev_set_sampler_voices,
        shuffle = False
    )

    eval_loader_voices = td.DataLoader(
        eval_set_voices,
        batch_size=args['batch_size']//4,
        pin_memory=True,
        num_workers=args['num_workers']//2,
        sampler=eval_set_sampler_voices,
        shuffle = False
    )
    
    return train_set_sampler, train_loader, test_set, test_loader, dev_loader_voices, eval_loader_voices

class TrainSet(td.Dataset):
    def __init__(self, args, dataset):
        self.items = dataset
        self.args = args
        
        # set label
        count = 0

        # crop size
        self.crop_size = args['winlen'] + (args['winstep'] * (args['train_frame'] - 1))

        self.p = args['DA_p']
    
        # musan
        self.noises = Noises(f'{args["path_musan"]}_split/train')
        
        # rir
        self.rir = RIRReverberation(args['path_rir'])

        # aug
        self.apply_augmentation = Compose(
            transforms=[
                Gain(
                    min_gain_in_db=-15.0,
                    max_gain_in_db=5.0,
                    p=0.2,
                ),
                PitchShift(p=0.2, sample_rate=16000),
                AddColoredNoise(p=0.2)
            ]
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)
        
        # rand crop
        audio = torch.from_numpy(audio).float()
        audio = util.rand_crop(audio, self.crop_size)

        if self.p < random.random():
            return audio, audio, item.label, False
        else:
            ref_audio = copy.deepcopy(audio)
            if len(self.args['gpu_ids']) >1:
                audio = self.apply_augmentation(audio.unsqueeze(0).unsqueeze(0).to(self.args['device']), sample_rate=16000).squeeze(0).squeeze(0).cpu()
            else:
                audio = self.apply_augmentation(audio.unsqueeze(0).unsqueeze(0), sample_rate=16000).squeeze(0).squeeze(0)
            aug_type = random.randint(0, 4)
            if aug_type == 0:
                audio = self.rir(audio)
            elif aug_type == 1:
                audio = self.noises(audio, ['speech'])
            elif aug_type == 2:
                audio = self.noises(audio, ['music'])
            elif aug_type == 3:
                audio = self.noises(audio, ['noise'])
            elif aug_type == 4:
                audio = self.noises(audio, ['speech', 'music'])

            return audio, ref_audio, item.label, True
               
        
class TestSet(td.Dataset):
    @property
    def Key(self):
        return self.key
    @Key.setter
    def Key(self, value):
        self.key = value

    def __init__(self, args, dataset):
        self.key = 'clean'
        self.items = dataset
        self.crop_size = args['winlen'] + (args['winstep'] * (args['test_frame'] - 1))
        self.num_seg = args['num_seg']
        
    def __len__(self):
        return len(self.items[self.Key])

    def __getitem__(self, index):
        item = self.items[self.Key][index]

        # read wav
        audio, _ = sf.read(item.path)

        # crop
        if audio.shape[0] <= self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        # stack
        buffer = []
        indices = np.linspace(0, audio.shape[0] - self.crop_size, self.num_seg)
        for idx in indices:
            idx = int(idx)
            buffer.append(audio[idx:idx + self.crop_size])
        buffer = np.stack(buffer, axis=0)

        return buffer.astype(np.float), item.key

class TestSet_voices(td.Dataset):
    @property
    def Key(self):
        return self.key
    @Key.setter
    def Key(self, value):
        self.key = value

    def __init__(self, args, dataset):
        self.items = dataset
        self.crop_size = args['winlen'] + (args['winstep'] * (args['test_frame'] - 1))
        self.num_seg = args['num_seg']
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        # read wav
        audio, _ = sf.read(item.path)

        # crop
        if audio.shape[0] <= self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        # stack
        buffer = []
        indices = np.linspace(0, audio.shape[0] - self.crop_size, self.num_seg)
        for idx in indices:
            idx = int(idx)
            buffer.append(audio[idx:idx + self.crop_size])
        buffer = np.stack(buffer, axis=0)

        return buffer.astype(np.float), item.key

def round_down(num, divisor):
    return num - (num % divisor)