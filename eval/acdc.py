import torch.nn.functional as F
import torch
import sys
import os
from vision_transformer import get_dino_backbone
from universeg import universeg
import numpy as np
from example_Data.acdc import ACDCDataset
from example_Data.SCD import SCDDataset
from example_Data.spine import SPINEDataset
from example_Data.pandental import PanDataset
from example_Data.wbc import WBCDataset
from example_Data.stare import StareDataset
from example_Data.monuseg import MonusegDataset
from example_Data.hipxray import HipXrayDataset
from example_Data.cervix import CervixDataset
import itertools
import math
import matplotlib.pyplot as plt
import einops as E
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from EICSeg import MamICL
from PIL import Image
import os
import imgviz
import cv2
from peft import PeftModel, PeftConfig
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
import torch
import time
from visual_feature import get_feature
from matcher.Matcher_SemanticSAM import build_matcher_oss as build_matcher_semantic_sam_oss

color_map = {
    0: (160, 232, 130),
    1: (226, 226, 231),
    2: (243, 213, 211),
    3: (209, 244, 224),
    4: (255, 242, 204),
    5: (206, 175, 95),
    6: (148, 72, 140),
    7: (110, 87, 193),
    8: (41, 147, 107),
    9: (25, 91, 141)
}

from scipy.spatial import distance

def replace_with_closest_label(image, color_map):
    """
    将RGB图像中的每个像素替换为color map中颜色最接近的label id
    
    参数:
        image: numpy数组，形状为(H, W, 3)的RGB图像
        color_map: 字典，格式为{label_id: (R, G, B)}
    
    返回:
        label_image: numpy数组，形状为(H, W)，包含最接近的label id
    """
    # 将color map转换为numpy数组
    colors = np.array(list(color_map.values()))
    labels = np.array(list(color_map.keys()))
    
    # 获取图像尺寸并重塑为像素列表
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    
    # 计算每个像素到所有color map颜色的距离
    dists = distance.cdist(pixels, colors, 'euclidean')
    
    # 找到最接近的label id
    closest_idx = np.argmin(dists, axis=1)
    label_image = labels[closest_idx].reshape(h, w)
    
    return label_image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=475):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=675, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=675, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))




def save_colored_mask_1(mask, save_path):
    h, w = mask.shape
    save_mask = np.zeros((h, w, 3))
    print(np.unique(mask))
    for i in np.unique(mask):
        if i != 0:
            save_mask[mask == i] = color_map[i]
    # cv2.imwrite(save_path, save_mask)
    return save_mask

def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)

    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/data1/output_dir_simple_1/')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='/dataset/zhongqiaoyong/MedicalImages/preprocess_rgb/imgs/bus/benign (106).npy')
    parser.add_argument('--input_size', type=int, default=448)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--conf_files',
                        default="/newdata3/xsa/ICUSeg/configs/seem_dino_lang.yaml",
                        metavar="FILE",
                        help='path to config file', )
    parser.add_argument('--support_size', default=2)
    parser.add_argument('--model_choice', default='ours')
    parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)

     # Dataset parameters
    parser.add_argument('--datapath', default='/newdata3/xsa/')
    parser.add_argument('--benchmark', default='fss',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', default=0)
    parser.add_argument('--nshot', default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', default='output/coco/fold0')
    parser.add_argument('--visualize', type=int, default=0)

    # DINOv2 and SAM parameters
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', default="/newdata3/xsa/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="/newdata3/xsa/sam_vit_h_4b8939.pth")
    parser.add_argument('--use_semantic_sam', action='store_true', help='use semantic-sam')
    parser.add_argument('--semantic-sam-weights', type=str, default="/newdata3/xsa/swint_only_sam_many2many.pth")
    parser.add_argument('--points_per_side', type=int, default=64)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sel_stability_score_thresh', default=0.9)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--iou_filter', default=0.85)
    parser.add_argument('--box_nms_thresh', default=0.65)
    parser.add_argument('--output_layer', type=int, default=3)
    parser.add_argument('--dense_multimask_output', type=int, default=0)
    parser.add_argument('--use_dense_mask', default=1)
    parser.add_argument('--multimask_output', default=1)

    # Matcher parameters
    parser.add_argument('--num_centers', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_box', action='store_true', help='use box as an extra prompt for sam')
    parser.add_argument('--use_points_or_centers', action='store_true', help='points:T, center: F')
    parser.add_argument('--sample-range', default=[1, 6], help='sample points number range')
    parser.add_argument('--max_sample_iterations', default=64)
    parser.add_argument('--alpha', default=1.)
    parser.add_argument('--beta', default=0.)
    parser.add_argument('--exp', default=0.)
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--purity_filter', default=0.02, help='use purity_filter')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--use_score_filter', default=True)
    parser.add_argument('--deep_score_norm_filter', type=float, default=0.1)
    parser.add_argument('--deep_score_filter', type=float, default=0.33)
    parser.add_argument('--topk_scores_threshold', default=0.0)
    parser.add_argument('--num_merging_mask', default=9, help='topk masks for merging')

    return parser.parse_args()


def visualize_tensors(tensors, col_wrap=8, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N / cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d * cols, d * rows))
    if rows == 1:
        axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M * (k // cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row, col]
            if len(x.shape) == 2:
                ax.imshow(x, vmin=0, vmax=1, cmap='gray')
            else:
                ax.imshow(E.rearrange(x, 'C H W -> H W C'))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig("{}.png".format(title))
    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()


def dice_score(predict, target):
    if torch.is_tensor(predict):
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))  #转一维数组
    target = np.atleast_1d(target.astype(np.bool))

    intersection = np.count_nonzero(predict & target) #计算非零个数

    size_i1 = np.count_nonzero(predict)
    size_i2 = np.count_nonzero(target)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0

    return dice

@torch.no_grad()
def inference_seggpt(model, image, label_onehot, support_images, support_labels_onehot, device):
    from Painter.SegGPT.SegGPT_inference.seggpt_engine import run_one_image

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    out_list = []
    size = image.size
    res, hres = 448, 448

    n_labels = label_onehot.shape[0]  # (4, 448, 448)
    image = Image.fromarray((image[0].cpu().numpy()* 255).astype(np.uint8)).convert('RGB')
    # image = Image.open("/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/CT_SPINE_case8-028.jpg").convert('RGB')
    image = image.resize((res, res))
    input_image = np.array(image) / 255

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(support_images, support_labels_onehot):
        img2 = Image.fromarray(img2_path[0].cpu().numpy() * 255).convert('RGB')
        # img2 = Image.open("/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/CT_SPINE_case8-031.jpg").convert('RGB')
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        # tgt2 = Image.open("/newdata3/xsa/ICUSeg/mambamodel/eval/visual/modified_image2.png").convert("RGB")
        tgt2_path = save_colored_mask_1(tgt2_path, None)
        tgt2 = Image.fromarray(np.uint8(tgt2_path)).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, input_image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    time_start = time.time()
    output = run_one_image(img, tgt, model, device)
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum
    output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[res, res],
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()

    out_list.append(torch.tensor(output.astype(np.uint8)))
    output = output.astype(np.uint8)
    label_image_1 = replace_with_closest_label(output, color_map)
    scores = []
    for k in np.unique(label_onehot):
        if k != 0:
            score = dice_score(label_image_1==k, label_onehot==k)
            scores.append(score)
            print(score)
    
    return {'Image': image,
            'Soft Prediction': [],
            'Prediction': [],
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}

@torch.no_grad()
def inference_multi_our(model, image, label_onehot, support_images, support_labels_onehot, device, name):
    label_onehot = label_onehot[1:]
    support_labels_onehot = support_labels_onehot[:, 1:]

    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)

    support_size, _, h, w = support_images.shape
    image = (image - image.min()) / (image.max() - image.min())
    support_images = (support_images - support_images.min()) / (support_images.max() - support_images.min())

    train_img = image[None].repeat(1, 3, 1, 1)
    support_images = support_images.repeat(1, 3, 1, 1)
    ref_masks = torch.zeros((support_size, 10, h, w))
    ref_masks[:, :n_labels, :, :] = support_labels_onehot[:, :n_labels, :, :]
    
    time_start = time.time()
    for _ in range(1):
        logits = model(
            train_img,
            label_onehot,
            support_images,
            ref_masks, mode='test', name=name
        )
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum
    torch.cuda.empty_cache()
    soft_pred = torch.sigmoid(logits)
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)

    scores = []
    for k in range(n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print('ours:{}'.format(score))
    
    save_mask = torch.zeros((h, w), dtype=torch.uint8)
    for id in range(n_labels):
        save_mask[hard_pred_1[id][0]] = torch.tensor(id + 5, dtype=torch.uint8)
        mask = hard_pred_1[id][0]
        mask[hard_pred_1[id][0] == 1] = torch.tensor(id + 5, dtype=torch.uint8)

    backmask = label_onehot.sum(dim=0) == 0
    label_onehot_save = torch.argmax(label_onehot, dim=0) + 20
    label_onehot_save[backmask] = 0

    return {'Image': image,
            'Soft Prediction': soft_pred_onehot,
            'Prediction': hard_pred_1,
            'Ground Truth': label_onehot,
            'score': np.mean(scores),
            'save_mask':save_mask,
            'label_onehot_save':label_onehot_save}

@torch.no_grad()
def inference_multi_matcher(model, image, label_onehot, support_images, support_labels_onehot, name, device):
    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)

    image = F.interpolate(image.unsqueeze(0), (518, 518), align_corners=True, mode='bilinear').squeeze(0)
    label_onehot = F.interpolate(label_onehot.unsqueeze(0), (518, 518), mode='nearest').squeeze(0)
    support_labels_onehot = F.interpolate(support_labels_onehot, (518, 518), mode='nearest').to(device)
    support_images = F.interpolate(support_images, (518, 518), align_corners=True, mode='bilinear').to(device)

    support_size, _, h, w = support_images.shape

    soft_pred_onehot = []

    time_start = time.time()
    train_img = image[None].repeat(1, 3, 1, 1)
    support_images = support_images.repeat(1, 3, 1, 1)

    time_start = time.time()

    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        model.set_reference(support_images[None], support_labels)
        model.set_target(train_img)
        pred_mask = model.predict()
        soft_pred_onehot.append(pred_mask)
    
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum

    soft_pred_onehot = torch.stack(soft_pred_onehot)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)

    scores = []
    for k in range(1, n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print(score)

    return {'Image': image,
            'Soft Prediction': soft_pred_onehot,
            'Prediction': hard_pred_1,
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}

@torch.no_grad()
def inference_multi(model, image, label_onehot, support_images, support_labels_onehot, name, device):
    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)

    image = F.interpolate(image.unsqueeze(0), (128, 128), align_corners=True, mode='bilinear').squeeze(0)
    label_onehot = F.interpolate(label_onehot.unsqueeze(0), (128, 128), mode='nearest').squeeze(0)
    support_labels_onehot = F.interpolate(support_labels_onehot, (128, 128), mode='nearest').to(device)
    support_images = F.interpolate(support_images, (128, 128), align_corners=True, mode='bilinear').to(device)

    support_size, _, h, w = support_images.shape

    soft_pred_onehot = []

    time_start = time.time()

    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        logits = model(
            image[None],
            label_onehot,
            support_images[None],
            support_labels[None]
        )[0]
        soft_pred = torch.sigmoid(logits)
        soft_pred_onehot.append(soft_pred)
    
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum

    soft_pred_onehot = torch.stack(soft_pred_onehot)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)

    scores = []
    for k in range(1, n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print(score)

    return {'Image': image,
            'Soft Prediction': soft_pred_onehot,
            'Prediction': hard_pred_1,
            'Ground Truth': label_onehot,
            'score': np.mean(scores),
            'save_mask':save_mask,
            'label_onehot_save':label_onehot_save}

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    h, w = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

# hipxray: 47.7  57.1

def inference_multi_persam(model, image, label_onehot, support_images, support_labels_onehot, name):
    from PerSAM.per_segment_anything import sam_model_registry, SamPredictor
    sam_type, sam_ckpt = 'vit_b', '/newdata3/xsa/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    sam.eval()

    test_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')
    test_image = np.array(test_image)

    predictor = SamPredictor(sam)
    print("======> Obtain Location Prior" )
    final_masks_clss = []
    time_start = time.time()
    for l in range(1, support_labels_onehot.shape[1]):
        final_masks = []
        #对于每个类别，所有的ref_mask放在一起计算
        for k in range(support_images.shape[0]):
            ref_image_ori = Image.fromarray(np.uint8(support_images[k][0] * 255)).convert('RGB')
            ref_image_ori = np.array(ref_image_ori)

            ref_mask_ori = Image.fromarray(np.uint8(support_labels_onehot[k][l]))
            ref_mask_ori_1 = np.array(Image.fromarray(np.uint8(support_labels_onehot[k][l]) * 255).convert('RGB'))

            ref_mask = predictor.set_image(ref_image_ori, ref_mask_ori_1)
            ref_feat = predictor.features.squeeze().permute(1, 2, 0)

            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
            ref_mask = ref_mask.squeeze()[0]

            # Target feature extraction
            target_feat = ref_feat[ref_mask > 0]
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
            target_embedding = target_embedding.unsqueeze(0)


            predictor.set_image(test_image)
            test_feat_1 = predictor.features
            test_feat = test_feat_1.squeeze()
            # Cosine similarity
            C, h, w = test_feat.shape

            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
  
            sim = target_feat @ test_feat
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()

            # Positive-negative location prior
            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=20)
            topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

            # Obtain the target guidance for cross-attention layers
            sim = (sim - sim.mean()) / torch.std(sim)
            sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

            # First-step prediction
            masks, scores, logits, _, _ = predictor.predict(
                point_coords=topk_xy, 
                point_labels=topk_label, 
                multimask_output=False,
                attn_sim=attn_sim,  # Target-guided Attention
                target_embedding=target_embedding  # Target-semantic Prompting
            )
            best_idx = 0

            # Cascaded Post-refinement-1
            masks, scores, logits, _, upscaling = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits[best_idx: best_idx + 1, :, :], 
                        multimask_output=True)
            best_idx = np.argmax(scores)

            y, x = np.nonzero(masks[best_idx])
            if x.shape[0]!=0:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks, scores, logits, _ , _= predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                    multimask_output=True)
                best_idx = np.argmax(scores)

            # Save masks
            time_end = time.time()
            sim = F.interpolate(sim, (448, 448), align_corners=True, mode='bilinear')
            sim = sim.squeeze(0).squeeze(0).cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(sim)
        

            final_mask = masks[best_idx]
            final_masks.append(torch.tensor(final_mask))

        final_cls = torch.tensor(torch.stack(final_masks, dim=0),dtype=torch.float32)
        final_cls_avg = final_cls.mean(dim=0)
        final_masks_clss.append(final_cls_avg)

    final_masks_clss = torch.stack(final_masks_clss, dim=0)
    scores = []
    torch.cuda.synchronize()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum
    
    print(torch.unique(final_masks_clss))

    for k in range(1, support_labels_onehot.shape[1]):
        score = dice_score((final_masks_clss[k-1] > 0), label_onehot[k])
        scores.append(score)
        print('persam:{}'.format(score))
    
    save_mask = torch.zeros((1, 1, 448, 448), dtype=torch.uint8)
    for id in range(1, support_labels_onehot.shape[1]):
        save_mask[0, 0, final_masks_clss[id-1] > 0] = torch.tensor(id + 4, dtype=torch.uint8)

    save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    save_mask = save_mask.squeeze(0).squeeze(0)

    return {'Image': image,
            'Soft Prediction': final_masks_clss[k-1],
            'Prediction': final_masks_clss[k-1],
            'Ground Truth': label_onehot,
            'score': np.mean(scores),
            'save_mask':save_mask
            }


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)



args = get_args_parser()
model_univer = universeg(pretrained=True)
_ = model_univer.to('cuda')
model_univer.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)


def prepare_model_seggpt(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    import Painter.SegGPT.SegGPT_inference.models_seggpt as models_seggpt

    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


repetation = 10
total_univer = []
total_ours_3 = []
total_ours_1 = []
total_ours_2 = []


matcher = build_matcher_semantic_sam_oss(args)
d_support = ACDCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
d_test = ACDCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))


backbone = get_dino_backbone(opt).cuda()
cnt = 0
with torch.no_grad():
    for rep in range(repetation):
        n_support = 1
        support_images, support_labels, support_name = zip(*itertools.islice(d_support, n_support))
        print(support_name)

        support_images = torch.stack(support_images).to('cpu')
        support_labels = torch.stack(support_labels).to('cpu')
        
        n_viz = 16
        n_predictions = 100
        results_univer = defaultdict(list)
        results_ours = defaultdict(list)
        results_ours_1 = defaultdict(list)
        results_ours_2 = defaultdict(list)
        results_persam = defaultdict(list)

        idxs = np.random.permutation(len(d_test))[:n_predictions]

        for i in tqdm(idxs):
            cnt +=1
            image, label, name = d_test[i]
            vals_ours_2 = inference_multi_our(model_our_2, image, label, support_images, support_labels, 'cuda', name)

            for k, v in vals_ours_2.items():
                results_ours_1[k].append(v)

        scores_ours_1 = results_ours_1.pop('score')
        avg_score_ours_1 = np.mean(scores_ours_1)
        total_ours_1.append(avg_score_ours_1)

    avg_ours_1 = np.mean(total_ours_1)
    std_ours_1 = np.std(total_ours_1)

    print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours_1))
    print('Our std dice score after 5 repetations:{}'.format(std_ours_1))
    print(sum_time / 100)
