from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from utils.ddp_util import all_gather
import utils.metric as metric
from torch.autograd import Variable

def load_dict(args, model, dic):
    if len(args['gpu_ids']) <=1 and list(dic.keys())[0][:6] == 'module':
        new_dic = {}
        for key, value in zip(dic.keys(), dic.values()):
            new_dic[key[7:]] = value
        model.load_state_dict(new_dic)
    else: model.load_state_dict(dic)
    return model
    
class ModelTrainer:
    def __init__(self, args):
        args = None
        vox1 = None
        voices = None
        model = None
        logger = None
        criterion = None
        optimizer = None
        lr_scheduler = None
        train_loader = None
        enrollment_set = None
        enrollment_loader = None
        dev_loader_voices = None
        eval_loader_voices = None

    def run(self):
        self.best_eer = 1000
        self.update_best_model_dic = [0]
        self.best_model_dic = None
        self.best_epoch = None
        self.idx_for_log = len(self.train_loader) // self.args['number_iteration_for_log']
        self.log_step = 0
        self.global_step = 0
        self._loss = 0.
        self._loss_clf = 0.
        self._loss_prior = 0.
        self._loss_diff = 0.

        for epoch in range(self.args['epoch']):
            self.train_set_sampler.set_epoch(epoch)
            self.train(epoch)
            self.test(epoch)

        ### Activate to evaluate the model using pre-trained weights ###  
        # path_bestmodel =  '/workspace/Diff-SV/code/weight/diff-sv.pt'
        # self.best_model_dic = torch.load(path_bestmodel, map_location=self.args['device'])
        # self.model = load_dict(self.args, self.model, self.best_model_dic['model'])
        
        self.model.load_state_dict(self.best_model_dic)
        self.test_noisy()    

    def train(self, epoch):
        self.model.train()
        idx_ct_start = len(self.train_loader)*(int(epoch))
        with tqdm(total = len(self.train_loader), ncols = 200) as pbar:
            for idx, (m_batch, m_batch_referance, m_label, m_aug) in enumerate(self.train_loader):
                loss = 0
                self.optimizer.zero_grad()
                
                m_label = m_label.to(self.args['device'])
                m_aug = m_aug.to(self.args['device'])
                m_batch = m_batch.to(torch.float32).to(self.args['device'], non_blocking=True)
                m_batch_referance = m_batch_referance.to(torch.float32).to(self.args['device'], non_blocking=True)

                code, loss_prior, loss_diff = self.model(m_batch, m_batch_referance, m_aug, epoch, is_train=True)
                
                description = '%s epoch: %d '%(self.args['name'], epoch)
                
                loss_clf = self.criterion['classification_loss'](code, m_label)
                loss += loss_clf
                self._loss_clf += loss_clf.cpu().detach() 
                description += 'loss_clf:%.3f '%(loss_clf)
                
                loss += loss_prior
                self._loss_prior += loss_prior.cpu().detach() 
                description += 'loss_prior:%.3f '%(loss_prior)

                loss += loss_diff
                self._loss_diff += loss_diff.cpu().detach()
                description += 'loss_diff:%.3f '%(loss_diff)

                self._loss += loss.cpu().detach()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                loss.backward()
                self.optimizer.step()
    
                description += 'TOT: %.4f'%(loss)
                pbar.set_description(description)
                pbar.update(1)
                self.global_step +=1

                ###  if the current epoch is match to the logging condition, log ###
                if self.global_step % self.idx_for_log == 0:
                    self._loss /= self.idx_for_log
                    self._loss_clf /= self.idx_for_log
                    self._loss_diff /= self.idx_for_log
                    self._loss_prior /= self.idx_for_log

                    for p_group in self.optimizer.param_groups:
                        lr = p_group['lr']
                        break

                    if self.args['flag_parent']:
                        self.logger.log_metric('loss', self._loss, log_step = self.log_step)
                        self.logger.log_metric('loss_clf', self._loss_clf, log_step = self.log_step)
                        self.logger.log_metric('loss_diff', self._loss_diff, log_step = self.log_step)
                        self.logger.log_metric('loss_prior', self._loss_prior, log_step = self.log_step)
                        self.logger.log_metric('lr', lr, log_step = self.log_step)

                        self._loss = 0.
                        self._loss_clf = 0.
                        self._loss_diff = 0.
                        self._loss_prior = 0.
                        self.log_step += 1

                self.lr_scheduler.step()        


    def test(self, epoch):
        ### evaluate clean evaluation ###
        self.enrollment_set.Key = 'clean'
        self.embeddings = self._enrollment(self.enrollment_loader)
        if self.args['flag_parent']:
            self.cur_eer, min_dcf = self._calculate_eer(self.vox1.test_trials)
            self.logger.log_metric('EER_clean', self.cur_eer, epoch_step=epoch)
            self.logger.log_metric('Min_DCF_clean', min_dcf, epoch_step=epoch)
    
            if self.cur_eer < self.best_eer:
                self.best_eer = self.cur_eer
                self.logger.log_metric('BestEER_clean', self.best_eer, epoch_step=epoch)
                check_point = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
				    'lr_scheduler': self.lr_scheduler.state_dict(),
                }
                self.logger.save_model('Checkpoint_{}_{}'.format(epoch, self.best_eer), check_point)
                self.update_best_model_dic = [1]
        if len(self.args['gpu_ids']) >1: self._synchronize()

        self.update_best_model_dic = all_gather(self.update_best_model_dic)
        if sum(self.update_best_model_dic):
            self.best_model_dic = self.model.state_dict()
            self.best_epoch = epoch
            self.update_best_model_dic = [0]
    
    def test_noisy(self):
        ### 1. in-domain speech evaluation with in-domain noise ###
        self.test_noise_set(self.best_epoch, 'noise_0')
        self.test_noise_set(self.best_epoch, 'noise_5')
        self.test_noise_set(self.best_epoch, 'noise_10')
        self.test_noise_set(self.best_epoch, 'noise_15')
        self.test_noise_set(self.best_epoch, 'noise_20')
        
        self.test_noise_set(self.best_epoch, 'speech_0')
        self.test_noise_set(self.best_epoch, 'speech_5')
        self.test_noise_set(self.best_epoch, 'speech_10')
        self.test_noise_set(self.best_epoch, 'speech_15')
        self.test_noise_set(self.best_epoch, 'speech_20')
        
        self.test_noise_set(self.best_epoch, 'music_0')
        self.test_noise_set(self.best_epoch, 'music_5')
        self.test_noise_set(self.best_epoch, 'music_10')
        self.test_noise_set(self.best_epoch, 'music_15')
        self.test_noise_set(self.best_epoch, 'music_20')

        ### 2. in-domain speech evaluation with out-of-domain noise ###
        self.test_noise_set(self.best_epoch, 'nonspeech_0')
        self.test_noise_set(self.best_epoch, 'nonspeech_5')
        self.test_noise_set(self.best_epoch, 'nonspeech_10')
        self.test_noise_set(self.best_epoch, 'nonspeech_15')
        self.test_noise_set(self.best_epoch, 'nonspeech_20')

        ### 3. out-of-domain speech evaluation with out-of-domain noise ###
        self.test_voices(self.best_epoch, 'voices_dev', self.dev_loader_voices, self.voices.dev_trials)
        self.test_voices(self.best_epoch, 'voices_eval', self.eval_loader_voices, self.voices.eval_trials)

    def test_noise_set(self, epoch, key):
        self.enrollment_set.Key = key
        self.embeddings = self._enrollment(self.enrollment_loader)
        if self.args['flag_parent']:
            eer, min_dcf = self._calculate_eer(self.vox1.test_trials)
            self.logger.log_metric(f'EER_{key}', eer, epoch_step=epoch)
            self.logger.log_metric(f'Min_DCF_{key}', min_dcf, epoch_step=epoch)
        if len(self.args['gpu_ids']) >1: self._synchronize()

    def test_voices(self, epoch, key, loader, trial):
        self.embeddings = self._enrollment(loader)
        if self.args['flag_parent']:
            eer, min_dcf = self._calculate_eer(trial)
            self.logger.log_metric(f'EER_{key}', eer, epoch_step=epoch)
            self.logger.log_metric(f'Min_DCF_{key}', min_dcf, epoch_step=epoch)
        if len(self.args['gpu_ids']) >1: self._synchronize()

    def _enrollment(self, enrollment_loader):
        ### Return embedding dictionary ###
        self.model.eval()

        keys = []
        embeddings = []
        
        with tqdm(total=len(enrollment_loader), ncols=90) as pbar, torch.set_grad_enabled(False):
            if len(self.args['gpu_ids']) >1:
                with self.model.no_sync():
                    for x_seg, key in enrollment_loader:

                        x_seg = x_seg.to(torch.float32).to(self.args['device'], non_blocking=True).view(-1, x_seg.size(-1)) 
                        x_seg = self.model(x_seg, is_train=False).to('cpu').view(-1, self.args['num_seg'], self.args['code_dim']) 

                        keys.extend(key)
                        embeddings.extend(x_seg)

                        if self.args['flag_parent']:
                            pbar.update(1)
                            
                keys = all_gather(keys)
                embeddings = all_gather(embeddings)
            else:
                for x_seg, key in enrollment_loader:

                    x_seg = x_seg.to(torch.float32).to(self.args['device'], non_blocking=True).view(-1, x_seg.size(-1)) 
                    x_seg = self.model(x_seg, is_train=False).to('cpu').view(-1, self.args['num_seg'], self.args['code_dim']) 

                    keys.extend(key)
                    embeddings.extend(x_seg)
                    pbar.update(1)

        seg_dict = {}
        for i in range(len(keys)):
            seg_dict[keys[i]] = embeddings[i]

        return seg_dict
    
    def _calculate_eer(self, test_trials):
        labels = []
        cos_sims = [[], []]

        for item in tqdm(test_trials, desc='test', ncols=self.args['tqdm_ncols']):
            cos_sims[0].append(self.embeddings[item.key1])
            cos_sims[1].append(self.embeddings[item.key2])
            labels.append(int(item.label))
        
        batch = len(labels)
        num_seg = self.args['num_seg']
        buffer1 = torch.stack(cos_sims[0], dim=0).view(batch, num_seg, -1)
        buffer2 = torch.stack(cos_sims[1], dim=0).view(batch, num_seg, -1)
        buffer1 = buffer1.repeat(1, num_seg, 1).view(batch * num_seg * num_seg, -1)
        buffer2 = buffer2.repeat(1, 1, num_seg).view(batch * num_seg * num_seg, -1)
        cos_sims = F.cosine_similarity(buffer1, buffer2)
        cos_sims = cos_sims.view(batch, num_seg * num_seg)
        cos_sims = cos_sims.mean(dim=1)

        eer = metric.calculate_EER(
            scores=cos_sims, labels=labels
        )
        min_dcf = metric.calculate_MinDCF(
            scores=cos_sims, labels=labels
        )
        return eer, min_dcf

    def _synchronize(self):
        torch.cuda.empty_cache()
        dist.barrier()