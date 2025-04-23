from typing import Optional
import sys
import imgviz
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from PIL import Image
import numpy as np
from cluster import CTM
import matplotlib.pyplot as plt
from modeling.mask_decoder import MaskDecoder
from modeling.transformer import TwoWayTransformer
from typing import Any, Dict, List, Tuple
from modeling.prompt_encoder import PositionEmbeddingRandom
from sinkhorn_distance import Sinkhorn
all_dict = {}


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道提取空间信息
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 经过卷积提取空间注意力权重
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        # 输出非负
        out = self.sigmoid(out)
        return out


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


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class EICSegDecoder(nn.Module):
    def __init__(
            self):

        super().__init__()
        self.num_layers = 3
        self.num_feature_levels = 3
        self.final_fuse = nn.Sequential(
            nn.Conv2d(1536, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
              
        self.channel_reduction = nn.ModuleList()
        self.channel_reduction.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1),
                nn.BatchNorm2d(512)
            ))
        self.channel_reduction.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1),
                nn.BatchNorm2d(512)
            ))

        self.spatial_list = [64, 64, 32]
        self.channel_list = [512, 512, 1024]

        self.skip_connection = nn.ModuleList()
        self.skip_connection.append(
            nn.Sequential(
                nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
                nn.BatchNorm2d(self.channel_list[1])
            ))
        self.skip_connection.append(
            nn.Sequential(
                nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
                nn.BatchNorm2d(self.channel_list[2])
            )
        )

        self.CTM_module_list = nn.ModuleList()
        self.CTM_module_list.append(CTM(sample_ratio=256, dim_out=512, k=5))
        self.CTM_module_list.append(CTM(sample_ratio=128, dim_out=512, k=3))
        self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=1024, k=3))

        self.sam_decoder = MaskDecoder(
            num_multimask_outputs=10,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=1024,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.Convs = nn.Sequential(
            nn.Conv2d(10, 256 // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(256 // 4),
            nn.GELU(),
            nn.Conv2d(256 // 4, 256 // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(256 // 2),
            nn.GELU(),
            nn.Conv2d(256 // 2, 256, kernel_size=1),
        )

        self.pe_layer = PositionEmbeddingRandom(128)
        self.ln = nn.LayerNorm(512)
        self.spatial_atten = SpatialAttention()
        self.sink = Sinkhorn().eval()

    def forward(self, ref_information, query_information, mode, fea_sam=None, alpha=1, name=None):
        query_multi_scale = query_information
        ref_multiscale_feature, ref_mask = ref_information
        out_predict_list = []

        bs_sp, c, h, w = ref_mask.shape

        ref_mask_list = []
        for i in range(self.num_feature_levels):
            ref_mask_si = F.interpolate(ref_mask, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
            ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))


        query_stage_list = []
        ref_stage_list = []
        src_copy = []

        for i in range(self.num_feature_levels):
            if i != 2:
                query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
                                               (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                               mode='bilinear')
            else:
                query_multi_si = query_multi_scale[i]

            query_stage_list.append(query_multi_si)
            src_copy.append(query_multi_si.clone())

        for i in range(self.num_feature_levels):
            if i != 2:
                ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
                                             (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                             mode='bilinear')
            else:
                ref_multi_si = ref_multiscale_feature[i]

            ref_stage_list.append(ref_multi_si)

        spatial_cens = []
        spatial_params = []

        for level_index in range(self.num_feature_levels):
            if level_index != 0:
                pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
                                             mode='bilinear')
                query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
                query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])

            src_mask_features = query_stage_list[level_index]
            bs_src, d, _, _ = src_mask_features.shape

            if mode != 'test':
                spatial_tokens = ref_stage_list[level_index]
                bs_sp, d, _, _ = spatial_tokens.shape
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)

            else:
                spatial_tokens = ref_stage_list[level_index]
                bs_sp, d, _, _ = spatial_tokens.shape
        
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
                ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)

                token_dict = {'x': spatial_tokens,
                              'token_num': spatial_tokens.size(1),
                              'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
                                  spatial_tokens.size(0), 1),
                              'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
                              'mask': None,
                              'ref_mask': ref_mask}

                token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
                spatial_tokens = token_dict_down['x']
                temp_mask = token_dict_down['ref_mask']
                
            spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)

            spatial_cens.append(spatial_tokens)
            if mode == 'test':
                spatial_params.append(temp_mask)
            pre_feature = query_stage_list[level_index]

        if mode != 'test':
            for i in range(len(query_stage_list)):
                src_mask_features = query_stage_list[i]
                spatial_tokens = ref_stage_list[i]

                bs_src, d, _, _ = src_mask_features.shape
                bs_sp, d, _, _ = spatial_tokens.shape

                src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)

                src_norm = src_mask_features
                spatial_norm = spatial_tokens

                avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
                avg_atten = avg_atten.softmax(dim=-1)

                out_predict = avg_atten @ torch.tensor(ref_mask_list[i], dtype=torch.float32)
                out_predict_list.append(out_predict)

        else:
            for i in range(len(query_stage_list)):
                src_mask_features = query_stage_list[i]
                spatial_tokens = spatial_cens[i]

                bs_src, d, _, _ = src_mask_features.shape
                _, nums, _ = spatial_tokens.shape

                ref_mask = spatial_params[i].reshape(1, -1, 10)
                src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)

                src_norm = src_mask_features
                spatial_norm = spatial_tokens

                avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
                avg_atten_save = avg_atten
                avg_atten = avg_atten.softmax(dim=-1)

              
                topalpha = 1/2
                cost_matrix = 1 - avg_atten.reshape(src_norm.shape[1], bs_sp, -1).permute(1, 0, 2)
                P = self.sink(cost_matrix)
                dist = (P * cost_matrix).sum(-1).sum(-1)

                ###### 根据distance选出一定比例的in-context prompts
                real_num = int(bs_sp * topalpha)
                if real_num < 1:
                    real_num = 1
                    _, Indices = torch.topk(dist, bs_sp - real_num)
                    avg_atten_1 = avg_atten_save.reshape(src_norm.shape[1], bs_sp, -1)
                    avg_atten_1[:, Indices, :] = float('-inf')
                    avg_atten_1 = avg_atten_1.reshape(1, src_norm.shape[1], -1)
                    avg_atten = avg_atten_1.softmax(dim=-1)

                out_predict = avg_atten @ ref_mask
                out_predict_list.append(out_predict)

        if mode == 'test':
            rec_mask = spatial_params[0].reshape(1, -1, 10)
            rec_tokens = spatial_cens[0]
        else:
            rec_mask = ref_mask_list[0]
            bs_sp, d1, h1, w1 = ref_stage_list[0].shape
            rec_tokens = ref_stage_list[0].view(bs_sp, d1, -1).permute(0, 2, 1)

        results = self.forward_prediction_heads(src_copy, out_predict_list, mode=mode, fea_sam=fea_sam, ref_mask=rec_mask, spatial_tokens=rec_tokens, name=name)
        return results



    def forward_prediction_heads(self, src, out_predict_list, mode=None, fea_sam=None, ref_mask=None, spatial_tokens=None, name=None):
        bs, dim1, h1, w1 = src[0].shape
        bs, dim2, h2, w2 = src[1].shape
        bs, dim3, h3, w3 = src[2].shape

        f_1_aug = src[0]
        f_3_aug = src[2]
        f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
        out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
                                      align_corners=True)

        final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))  ### dino特征加载完毕
                                                                            ### dino的特征做q,k sam对应的特征图做v/256
        final_fuse_1 = self.spatial_atten(final_fuse) * fea_sam + fea_sam

        out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
        outputs_feature = self.Convs(
            out_predict
        )

        curr_embedding = outputs_feature + final_fuse_1

        low_res_masks, iou_predictions, upscaling = self.sam_decoder(
            image_embeddings=curr_embedding,
            multimask_output=True,
            image_pe=self.pe_layer.forward((64, 64)).unsqueeze(0),
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=None,
            target_embedding=None
        )

        masks = postprocess_masks(
            low_res_masks,
            input_size=(448, 448),
            original_size=(448, 448),
        )

        results = {
            "predictions_mask": masks
        }
        return results


def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
      masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
      input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
      original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
      (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    masks = F.interpolate(
        masks,
        input_size,
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks
