import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .trainer import Trainer
from models.ehr_models import EHR_encoder
from models.text_models import Text_encoder
from models.loss import CosineLoss
import os

class ContrastiveEHRRRTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl):
        super().__init__(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_encoder = EHR_encoder(args).to(self.device)
        self.text_encoder = Text_encoder(args, self.device).to(self.device)

        self.optimizer = optim.Adam(list(self.ehr_encoder.parameters()) + list(self.text_encoder.parameters()), args.lr, betas=(0.9, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5, mode='min')

        self.contrastive_loss = CosineLoss()

    def train_epoch(self):
        self.ehr_encoder.train()
        self.text_encoder.train()
        total_loss = 0

        for i, (x, _, _, rr, _, _, seq_lengths, _, _, _, _, _) in enumerate(self.train_dl):
            x = torch.from_numpy(x).float().to(self.device)
            rr = rr
            seq_lengths = seq_lengths.to(self.device)

            ehr_feats = self.ehr_encoder(x, seq_lengths)
            _, rr_feats = self.text_encoder(rr_notes=rr)

            loss = self.contrastive_loss(ehr_feats, rr_feats.mean(dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dl)
        return avg_loss

    def validate(self):
        self.ehr_encoder.eval()
        self.text_encoder.eval()
        total_loss = 0

        with torch.no_grad():
            for i, (x, _, _, rr, _, _, seq_lengths, _, _, _, _, _) in enumerate(self.val_dl):
                x = torch.from_numpy(x).float().to(self.device)
                rr = rr
                seq_lengths = seq_lengths.to(self.device)

                ehr_feats = self.ehr_encoder(x, seq_lengths)
                _, rr_feats = self.text_encoder(rr_notes=rr)

                loss = self.contrastive_loss(ehr_feats, rr_feats.mean(dim=1))
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dl)
        self.scheduler.step(avg_loss)
        return avg_loss

    def save_checkpoint(self, epoch, path='checkpoints/'):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'ehr_encoder_state_dict': self.ehr_encoder.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }, os.path.join(path, f'contrastive_checkpoint_epoch_{epoch}.pth'))
