import torch
import torch.nn as nn
from .classifier import Classifier  # Import from classifier.py
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import timm
import random

from .ehr_models import EHR_encoder
from .CXR_models import CXR_encoder
from .text_models import Text_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Fusion(nn.Module):
    def __init__(self, args, ehr_model=None, cxr_model=None, text_model=None):
        super(Fusion, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model
        self.text_model = text_model

        # Validate modalities in args
        assert hasattr(args, "modalities"), "args.modalities is required"
        self.modalities = args.modalities

        # Initialize based on fusion type
        fusion_type = args.fusion_type
        if fusion_type == 'unimodal_ehr':
            self.fusion_model = UnimodalEHR(args, ehr_model)
        elif fusion_type == 'unimodal_rr':
            self.fusion_model = UnimodalRR(args, text_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, *args, **kwargs):
        return self.fusion_model(*args, **kwargs)
        
    def equalize(self):
        return self.fusion_model.equalize()




class UnimodalEHR(nn.Module):
    def __init__(self, args, ehr_model):
        super(UnimodalEHR, self).__init__()
        self.args = args

        # EHR Encoder
        self.ehr_model = ehr_model
        if not self.ehr_model:
            raise ValueError("EHR encoder must be provided for UnimodalEHR!")
            
        # Define a classifier for EHR
        self.ehr_classifier = Classifier(self.ehr_model.feats_dim, self.args)


    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if x is None:
            raise ValueError("EHR data (x) must be provided for UnimodalEHR!")

        # Pass data through the encoder
        ehr_feats = self.ehr_model(x, seq_lengths)

        # Generate predictions using the classifier
        output = self.ehr_classifier(ehr_feats)

        return {'unimodal_ehr': output, 'unified': output}

        
class UnimodalRR(nn.Module):
    def __init__(self, args, text_model):
        super(UnimodalRR, self).__init__()
        self.args = args

        # Text Encoder
        self.text_model = text_model
        if not self.text_model or 'RR' not in self.args.modalities:
            raise ValueError("Text encoder with 'RR' modality must be provided for UnimodalRR!")

        # Define a classifier for RR
        self.rr_classifier = Classifier(self.text_model.feats_dim_rr, self.args)

    def forward(self, x=None, seq_lengths=None, img=None, pairs=None,  rr=None, dn=None):
        if rr is None:
            raise ValueError("RR data (rr) must be provided for UnimodalRR!")
            
        _, rr_feats = self.text_model(rr_notes=rr)
        if self.args.use_cls_token == 'cls':
            # Use the first token (CLS token)
            rr_feats = rr_feats[:, 0, :]
        else:
            # Use mean pooling
            rr_feats = rr_feats.mean(dim=1)

        # Generate predictions using the RR classifier
        output = self.rr_classifier(rr_feats)

        return {'unimodal_rr': output, 'unified': output}

