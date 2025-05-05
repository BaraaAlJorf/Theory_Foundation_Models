from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.fusion import Fusion
from models.ehr_models import EHR_encoder
from models.CXR_models import CXR_encoder
from models.text_models import Text_encoder
from models.loss_set import Loss

from .trainer import Trainer
import pandas as pd
import os

import numpy as np
from sklearn import metrics
import wandb

class MSMA_Trainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl):
        
        super(MSMA_Trainer, self).__init__(args)
        self.epoch = 0 
        self.start_epoch = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.seed = 379647
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        self.ehr_encoder = None
        self.cxr_encoder = None
        self.text_encoder = None
        
        if 'EHR' in args.modalities and args.ehr_encoder is not None:
            self.ehr_encoder = EHR_encoder(args)
        if ('RR' in args.modalities) and args.text_encoder is not None:
            self.text_encoder = Text_encoder(args, self.device)

            
        self.model = Fusion(args, self.ehr_encoder, self.cxr_encoder, self.text_encoder).to(self.device)
        
        self.load_model(args)  
        
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        
        self.loss = Loss(args)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None
    
    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs, rr, dn)
            
            
            pred = output[self.args.fusion_type].squeeze()
            if 'c-unimodal' in self.args.fusion_type:
                if self.args.task == 'phenotyping':
                    y = y.unsqueeze(1).repeat(1, pred.shape[1], 1)
                else:
                    y = y.unsqueeze(1).repeat(1, pred.shape[1])
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            if self.args.align > 0.0:
                loss = loss + self.args.align * output['align_loss']
                epoch_loss_align = epoch_loss_align + self.args.align * output['align_loss'].item()
            
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f} loss align {epoch_loss_align/i:0.4f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
                
        return ret
        
    def validate(self, dl, test):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0

        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, dn, rr, y_ehr, y_cxr, seq_lengths, pairs, age, gender, ethnicity, hadm_id) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                output = self.model(x, seq_lengths, img, pairs, rr, dn)
                
                
                pred = output[self.args.fusion_type]
                if self.args.fusion_type != 'uni_cxr':
                    if len(pred.shape) > 1:
                            pred = pred.squeeze()
            
                y = y.unsqueeze(1).repeat(1, pred.shape[1])
                
                            
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                if self.args.align > 0.0:
                    epoch_loss_align +=  output['align_loss'].item()
                    
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
        
        self.scheduler.step(epoch_loss/len(self.val_dl))
        
        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f} \t{epoch_loss_align/i:0.5f}")
        
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
            
        avg_loss = epoch_loss/i
        
        return ret, avg_loss
        
    def eval(self):
        #if self.args.mode == 'train':
        self.load_state(state_path=f'{self.args.save_dir}/{self.args.task}/{self.args.fusion_type}/best_checkpoint_{self.args.lr}_{self.args.task}_{self.args.fusion_type}_{self.args.modalities}_{self.args.data_pairs}.pth.tar')
        
        self.epoch = 0
        self.model.eval()
        
        ret, avg_loss = self.validate(self.test_dl, True)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename=f'results_{self.args.lr}_test.txt')
        return
    
    def train(self):
        test = False
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.model.eval()
            ret, avg_loss = self.validate(self.val_dl, False)
                
            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()
                print("checkpoint")
                self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                self.print_and_write(ret, isbest=False)
                self.patience+=1

            self.model.train()
            self.train_epoch()

            if self.patience >= self.args.patience:
                break
       
        self.print_and_write(self.best_stats , isbest=True)