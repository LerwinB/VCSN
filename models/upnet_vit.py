from mmengine.model import BaseModel
import torch.nn.functional as F
import torch.nn as nn
import torch
import logging
from mmengine import Config
from mmengine.logging import print_log
from torchvision.models.vision_transformer import vit_b_16
from mmseg.models.backbones import VisionTransformer
from mmseg.models.necks import MultiLevelNeck
from mmseg.models.decode_heads import UPerHead,FCNHead
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                        OptSampleList, SampleList, add_prefix)
from .encoder_decoder import MyEncoderDecoder
from mmseg.models.data_preprocessor import SegDataPreProcessor
class MMUpernetVit(BaseModel):

    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        data_preprocessor_dict = SegDataPreProcessor(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
            size = (512, 512))
        # data_preprocessor = Config(data_preprocessor_dict)
        self.backbone = VisionTransformer(
            img_size=(512, 512),
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic'
            )
        self.neck = MultiLevelNeck(
            in_channels=[768, 768, 768, 768],
            out_channels=768,
            scales=[4, 2, 1, 0.5])
        self.decode_head = UPerHead(
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            )
        self.auxiliary_head = FCNHead(
            in_channels=768,
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
            )
        self.encoder_decoder = MyEncoderDecoder(
            backbone=self.backbone,
            decode_head=self.decode_head,
            neck=self.neck,
            auxiliary_head=self.auxiliary_head,
            train_cfg=dict(),
            test_cfg=dict(mode='whole'),
            data_preprocessor=data_preprocessor_dict)

    

    
    def forward(self, imgs, data_samples=None, mode='tensor'):
        self.encoder_decoder(imgs,data_samples,mode)['out']
        
    
    