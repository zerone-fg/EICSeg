########################################################  current version ###############################################
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import imgviz
from PIL import Image
from modeling.mask_decoder import MaskDecoder
from modeling.transformer import TwoWayTransformer
from vision_transformer import get_dino_backbone
from modeling.prompt_encoder import PositionEmbeddingRandom
from utils.util import AverageMeter, count_params, init_log, DiceLoss
from focal_dice import FocalDiceLoss
import EIC_decoder
from ScribblePrompt.segment_anything.build_sam import sam_model_registry
from sam_loss import FocalDiceloss_IoULoss
from visual_feature import get_feature
all_dict = {}

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
 
 
def get_att_dis(target, behaviored):
    attention_distribution = []
    
    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].reshape(1, -1))  # 计算余弦相似度
        attention_distribution.append(attention_score)
    
    attention_distribution = torch.tensor(attention_distribution)
    indexes = torch.topk(attention_distribution, 32, largest=False).indices
    return indexes

def get_eu_dis(target, behaviored):
    attention_distribution = []
    pdist = nn.PairwiseDistance(p=2)
    for i in range(behaviored.size(0)):
        attention_score = pdist(target, behaviored[i].reshape(1, -1))  # 计算余弦相似度
        attention_distribution.append(attention_score)
    
    attention_distribution = torch.tensor(attention_distribution)
    indexes = torch.topk(attention_distribution, 32, largest=False).indices
    return indexes

class EICSeg(nn.Module):
    def __init__(self, cfg, strategy=None):
        super().__init__()
        self.backbone = get_dino_backbone(cfg)
        self.embed_dim = 512
        self.criterion = FocalDiceloss_IoULoss()
        self.decoder = EICdecoder()

        self.medencoder = sam_model_registry["vit_b"](checkpoint="checkpoint_sam.pt")
        self.medencoder.eval()
        for name, parms in self.medencoder.named_parameters():
            parms.requires_grad = False
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad)

    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs, targets, reference_img, ref_mask, cnt=0, mode="train", alpha=1):
        images = torch.tensor(batched_inputs)
        if mode == "train":
            ref_images = torch.tensor(reference_img).cuda()
            ref_masks = torch.tensor(ref_mask).cuda()  
        else:
            ref_images = torch.tensor(reference_img).cuda().unsqueeze(0)
            ref_masks = torch.tensor(ref_mask).cuda()

        b, shot, c, h, w = ref_images.shape
        assert not torch.isnan(ref_images.float()).any()
            
        features = self.backbone.get_intermediate_layers(images.float(), 3)  
        ref_features = self.backbone.get_intermediate_layers(ref_images.float().reshape(-1, c, h, w), 3) 
        ref_masks = ref_masks.float().reshape(-1, 10, h, w)
 
        with torch.no_grad():
            image_sam = F.interpolate(images.float(), (1024, 1024), mode='bilinear', align_corners=True)
            features_sam = self.medencoder.image_encoder(image_sam)
        
        features = [v for k, v in features.items()]
        ref_features = [v for k, v in ref_features.items()]
        ref_information = (ref_features, ref_masks)
        query_information = features

        outputs, _ = self.decoder(ref_information, query_information, mode=mode, fea_sam=features_sam, alpha=alpha)
        pred = F.interpolate(outputs["predictions_mask"], (targets.shape[-2], targets.shape[-1]), align_corners=True,
                            mode='bilinear')
        
        targets = targets.cuda()
        targets = targets.float()


        targets_1 = torch.argmax(targets, dim=1) 
        back_mask = (targets.sum(dim=1) == 0) 
        targets_1[back_mask] = 255

        losses = torch.tensor(0.0).cuda()
        cnt = 0

        if mode == "train":
            for id in range(10):
                losses += self.criterion(pred[:, id, :, :].unsqueeze(1), targets[:, id, :, :].unsqueeze(1))
            return losses
        else:
            return pred 

