import torch.nn.functional as F
import torch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('/newdata3/xsa/')
sys.path.append('/newdata3/xsa/UniverSeg-main')
sys.path.append('/newdata3/xsa/ICUSeg/mambamodel')
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
# from util.distributed import init_distributed
# from util.arguments import load_opt_from_config_files
# from xdecoder.BaseModel import BaseModel
# from xdecoder import build_model
# from CubeMamba import MamICL
from PIL import Image
import os
import imgviz
# from medpy.metric.binary import dc
import cv2
from peft import PeftModel, PeftConfig
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
import torch
import time
from visual_feature import get_feature
sum_time = 0
all_dict = {}

sys.path.append('/newdata3/xsa/Matcher-main')
from matcher.Matcher_SemanticSAM import build_matcher_oss as build_matcher_semantic_sam_oss

# a wbc
# b scd
# c acdc
# d:panden

#  140  72  148
# 95 175  206

color_map = {
    0: (160, 232, 130),
    1: (226, 226, 231),
    2: (243, 213, 211),
    3: (209, 244, 224),
    4: (255, 242, 204),
    5: (206, 175, 95),
    # # 5: (194, 65, 0),
    # 6: (148, 72, 140),
    # 7: (110, 87, 193),
    5: (64,128,128),
    # 7: (0, 64, 64),
    # 6: (128, 192, 128),
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


# def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
#     print(2*(y_pred*y_true).sum())
#     score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
#     return score.item()
# 28.5 7.4
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
    # label_image = Image.fromarray(output.astype(np.uint8))
    # label_image.save("/newdata3/xsa/ICUSeg/mambamodel/eval/visual/2.png")

    # size = image.size
    # res, hres = 448, 448
    # image = np.array(image.resize((res, hres))) / 255.

    # image_batch, target_batch = [], []
    # for l in range(label_onehot.shape[0]):
    #     if l == 0:
    #         continue
    #     for img2_, tgt2_ in zip(support_images, support_labels_onehot):
    #         img2 = Image.fromarray(img2_[0].cpu().numpy() * 255).convert('RGB')
    #         img2 = img2.resize((res, hres))
    #         img2 = np.array(img2) / 255.

    #         # tgt2 = Image.fromarray(tgt2_[0].cpu().numpy()).convert('L')
    #         # tgt2 = tgt2.resize((res, hres), Image.NEAREST)
    #         # tgt2 = np.array(tgt2)

    #         tgt2_test = Image.fromarray(np.uint8(tgt2_[l]) * 255).convert('RGB')
    #         # tgt2_test.save("1.png")

    #         tgt2 = np.array(tgt2_test) / 255.
    #         tgt = tgt2  # tgt is not available
    #         tgt = np.concatenate((tgt2, tgt), axis=0)
    #         img = np.concatenate((img2, image), axis=0)

    #         assert img.shape == (2 * res, res, 3), f'{img.shape}'
    #         # normalize by ImageNet mean and std
    #         img = img - imagenet_mean
    #         img = img / imagenet_std

    #         assert tgt.shape == (2 * res, res, 3), f'{img.shape}'
    #         # normalize by ImageNet mean and std
    #         tgt = tgt - imagenet_mean
    #         tgt = tgt / imagenet_std

    #         image_batch.append(img)
    #         target_batch.append(tgt)

    #     img = np.stack(image_batch, axis=0)
    #     tgt = np.stack(target_batch, axis=0)
    #     """### Run SegGPT on the image"""
    #     # make random mask reproducible (comment out to make it change)
    #     torch.manual_seed(2)
    #     output = run_one_image(img, tgt, model, device)
    #     output = F.interpolate(
    #         output[None, ...].permute(0, 3, 1, 2),
    #         size=[size[1], size[0]],
    #         mode='nearest',
    #     ).permute(0, 2, 3, 1)[0].numpy()

    #     out_list.append(torch.tensor(output.astype(np.uint8)))
    #     output = Image.fromarray(output.astype(np.uint8))
    #     output.save("2.png")

    # final_out = torch.stack(out_list, dim=0)
    # final_out = torch.argmax(final_out, dim=0)
    scores = []
    # label_onehot = F.interpolate(label_onehot, (res, res), mode='nearest')
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

# 229.6 ms的单张推理
# 587 ms的单张推理

# 882 ms
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
    # soft_pred = logits
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)
    # del soft_pred
    # del logits
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    # print(torch.sum(hard_pred_1))

    # hard_pred_1 = soft_pred_onehot
    # hard_pred_1 = F.softmax(10 * soft_pred_onehot, dim=0)

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


    # save_colored_mask_1(np.array(save_mask), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/acdc_vis/final_{}_{}.png'.format(id, np.mean(scores))))
    backmask = label_onehot.sum(dim=0) == 0
    label_onehot_save = torch.argmax(label_onehot, dim=0) + 20
    label_onehot_save[backmask] = 0
    # save_colored_mask_1(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/acdc_vis/label_{}_{}.png'.format(id, np.mean(scores))))

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

    # torch.cuda.synchronize()
    time_start = time.time()
    train_img = image[None].repeat(1, 3, 1, 1)
    support_images = support_images.repeat(1, 3, 1, 1)

    time_start = time.time()

    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        # logits = model(
        #     image[None],
        #     label_onehot,
        #     support_images[None],
        #     support_labels[None]
        # )[0]
        model.set_reference(support_images[None], support_labels)
        model.set_target(train_img)
        pred_mask = model.predict()
        # soft_pred = torch.sigmoid(logits)
        soft_pred_onehot.append(pred_mask)
    
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    global sum_time
    sum_time += time_sum
    # print(time_sum)

    soft_pred_onehot = torch.stack(soft_pred_onehot)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    # hard_pred_1 = soft_pred_onehot.round().clip(0,1)
    # soft_pred_onehot = torch.stack(soft_pred_onehot)
    # hard_pred_1 = F.softmax(10 * soft_pred_onehot, dim=0)
    # hard_pred_1 = torch.argmax(hard_pred_1, dim=0).squeeze(0)

    scores = []
    for k in range(1, n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print(score)

    # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
    # for id in range(1, n_labels):
    #     save_mask[0, 0, hard_pred_1[id][0] == 1] = torch.tensor(id + 4, dtype=torch.uint8)

    # save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    # save_mask = save_mask.squeeze(0).squeeze(0)
    # # save_colored_mask_1(np.array(save_mask), os.path.join('newnewenewnewnewnewfinal_uni{}_{}.png'.format(id, np.mean(scores))))

    # backmask = label_onehot[1:].sum(dim=0) == 0
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 4
    # label_onehot_save[backmask] = 0
    # save_colored_mask_1(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/visual/label_uni{}_{}_{}.png'.format(id, str(name).split("/")[-1], np.mean(scores))))

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

    # torch.cuda.synchronize()
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
    # print(time_sum)

    soft_pred_onehot = torch.stack(soft_pred_onehot)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    # hard_pred_1 = soft_pred_onehot.round().clip(0,1)
    # soft_pred_onehot = torch.stack(soft_pred_onehot)
    # hard_pred_1 = F.softmax(10 * soft_pred_onehot, dim=0)
    # hard_pred_1 = torch.argmax(hard_pred_1, dim=0).squeeze(0)

    scores = []
    for k in range(1, n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print(score)

    save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
    for id in range(1, n_labels):
        save_mask[0, 0, hard_pred_1[id][0] == 1] = torch.tensor(id + 4, dtype=torch.uint8)

    save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    save_mask = save_mask.squeeze(0).squeeze(0)
    # save_colored_mask_1(np.array(save_mask), os.path.join('newnewenewnewnewnewfinal_uni{}_{}.png'.format(id, np.mean(scores))))

    backmask = label_onehot[1:].sum(dim=0) == 0
    label_onehot_save = torch.argmax(label_onehot, dim=0) + 4
    label_onehot_save[backmask] = 0
    save_colored_mask_1(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/visual/label_uni{}_{}_{}.png'.format(id, str(name).split("/")[-1], np.mean(scores))))

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
    # from ScribblePrompt.segment_anything import sam_model_registry, SamPredictor
    # /newdata3/xsa/sam-med2d_b.pth
    # /newdata3/xsa/sam_vit_b_01ec64.pth
    # '/newdata3/xsa/sam_vit_b_01ec64.pth'
    sam_type, sam_ckpt = 'vit_b', '/newdata3/xsa/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    sam.eval()

    test_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')
    test_image = np.array(test_image)

    # ref_image_ori = Image.fromarray(np.uint8(support_images[0][0] * 255)).convert('RGB')
    # ref_image_ori = np.array(ref_image_ori)

    # ref_image = torch.tensor(ref_image_ori).cuda().unsqueeze(0)
    # ref_image = ref_image.permute(0, 3, 1, 2)
    # c, h, w = 3, 448, 448
    # ref_image = ref_image[None].repeat(1, 3, 1, 1)

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

            # from visual_feature import get_feature
            # all_dict["fea_sam"] = upscaling
            # get_feature(all_dict, "2", "2", name)

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
            # show_mask(masks[best_idx], plt.gca())
            # show_points(topk_xy, topk_label, plt.gca())
            # plt.title(f"Mask {best_idx}", fontsize=18)
            # plt.axis('off')
            vis_mask_output_path = os.path.join("/newdata3/xsa/", 'vis_mask_111newnew{}.jpg'.format(name))
            print(vis_mask_output_path)
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='jpg')

            final_mask = masks[best_idx]
            # mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            # mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            # mask_output_path = os.path.join("/newdata3/xsa/", "111" + '.png')
            # cv2.imwrite(mask_output_path, mask_colors)

            final_mask = masks[best_idx]
            # print(np.unique(final_mask))
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
    

    # predictor = SamPredictor(sam)
    # predictor.set_image(test_image)
    # test_feat_sam_1 = predictor.features
    # test_feat_sam = test_feat_sam_1.squeeze()
    # # Cosine similarity
    # C, h, w = test_feat_sam.shape

    # test_feat_sam = test_feat_sam / test_feat_sam.norm(dim=0, keepdim=True)
    # test_feat_sam = test_feat_sam.reshape(C, h * w).transpose(-1, -2)
    # # predictor.set_image(test_image)


    # images = torch.tensor(image).cuda()
    # train_img = images[None].repeat(1, 3, 1, 1)
    # features = model.get_intermediate_layers(train_img.float(), 3)  ##### 2, 1024, 32, 32

    # final_masks_clss = []
    # time_start = time.time()
    # for l in range(1, support_labels_onehot.shape[1]):
    #     final_masks = []
    #     #对于每个类别，所有的ref_mask放在一起计算
    #     for k in range(support_images.shape[0]):
    #         ref_image_ori = Image.fromarray(np.uint8(support_images[k][0] * 255)).convert('RGB')
    #         ref_image_ori = np.array(ref_image_ori)

    #         ref_image = torch.tensor(ref_image_ori).cuda().unsqueeze(0)
    #         ref_image = ref_image.permute(0, 3, 1, 2)
    #         c, h, w = 3, 448, 448
    #         # ref_image = ref_image[None].repeat(1, 3, 1, 1)

    #         ref_mask_ori = Image.fromarray(np.uint8(support_labels_onehot[k][l]))
    #         ref_mask_ori_1 = np.array(Image.fromarray(np.uint8(support_labels_onehot[k][l]) * 255).convert('RGB'))
    #         ref_mask = torch.tensor(np.array(ref_mask_ori)).cuda().unsqueeze(0).unsqueeze(0)
    #         ref_mask = F.interpolate(ref_mask, size=(64, 64), mode="nearest").float()

    #         ref_mask_sam = predictor.set_image(ref_image_ori, ref_mask_ori_1)
    #         ref_feat_sam = predictor.features.squeeze().permute(1, 2, 0)
    #         ref_mask_sam = F.interpolate(ref_mask_sam, size=ref_feat_sam.shape[0: 2], mode="bilinear")
    #         ref_mask_sam = ref_mask_sam.squeeze()[0]
    

    #         # ref_mask = F.interpolate(ref_mask, size=ref_feat_sam.shape[0: 2], mode="nearest")
    #         # ref_mask = ref_mask.squeeze()[0]

    #         # # Target feature extraction
    #         target_feat_sam = ref_feat_sam[ref_mask_sam > 0]
    #         target_embedding_sam = target_feat_sam.mean(0).unsqueeze(0)
    #         target_feat_sam = target_embedding_sam / target_embedding_sam.norm(dim=-1, keepdim=True)
    #         target_embedding_sam = target_embedding_sam.unsqueeze(0)

    #         # ref_features = model.get_intermediate_layers(ref_image.float().reshape(-1, c, h, w), 3) #### 10, 1024, 32, 32
    #         # fea_l = features['res3']
    #         # rea_l = ref_features['res3']

    #         # fea_l = F.interpolate(fea_l, (64, 64), mode='bilinear', align_corners=True).squeeze(0)
    #         # # fea_l = fea_l / fea_l.norm(dim=0, keepdim=True)
    #         # rea_l = F.interpolate(rea_l, (64, 64), mode='bilinear', align_corners=True).squeeze(0)
    #         # # rea_l = rea_l / rea_l.norm(dim=0, keepdim=True)

    #         # target_feat_dino = fea_l.reshape(1024, -1).transpose(-1, -2)
    #         # test_feat_dino = rea_l.reshape(1024, -1)
    #         ref_mask_sam = ref_mask.reshape(-1, 1)

    #         # sim = target_feat_dino @ test_feat_dino @ ref_mask_sam
    #         # sim = sim.reshape(1, 1, 64, 64)
    #         # sim_1 = sim.squeeze(0).squeeze(0)
    #         # plt.figure(figsize=(10, 10))
    #         # # out = sim.softmax(dim=-1)
    #         # plt.imshow(sim_1.cpu().detach().numpy())
    #         # plt.savefig("feature_{}.jpg".format(2),dpi=300)

    #         # # fea_l = F.interpolate(fea_l, (16, 16), mode='bilinear', align_corners=True)
    #         # # rea_l = F.interpolate(rea_l, (16, 16), mode='bilinear', align_corners=True)

    #         # rea_l = rea_l.squeeze(0).permute(1, 2, 0)
    #         # # fea_l = fea_l.reshape(1024, 32*32).mean(0).unsqueeze(0).transpose(-1, -2)
    #         # # rea_l = rea_l.reshape(1024, 32*32).mean(0).unsqueeze(0)
    #         # # sim = fea_l @ rea_l

    #         # ref_mask = F.interpolate(ref_mask, size=rea_l.shape[0: 2], mode="nearest")
    #         # ref_mask = ref_mask.squeeze()

    #         # target_feat = rea_l[ref_mask > 0]
    #         # target_embedding = target_feat.mean(0).unsqueeze(0)
    #         # target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    #         # target_embedding = target_embedding.unsqueeze(0)

    #         # print('======> Start Testing')

    #         # predictor.set_image(test_image)
    #         # test_feat = predictor.features.squeeze()
    #         # test_feat = fea_l.squeeze(0)
    #         # C, h, w = test_feat.shape
            
    #         # test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    #         # test_feat = test_feat.reshape(C, h * w)
    #         # print(torch.max(ref_mask_sam))
    #         target_feat_sam = ref_feat_sam.reshape(-1, 256).transpose(-1, -2)
    #         target_feat_sam = target_feat_sam / target_feat_sam.norm(dim=1, keepdim=True)

    #         # 
    #         # sim = test_feat_sam @ target_feat_sam @ ref_mask_sam
    #         sim = test_feat_sam @ target_feat_sam @ ref_mask_sam

    #         # sim = target_embedding_sam @ test_feat_sam @ ref_mask_sam
    #         # sim = (target_feat @ test_feat)

    #         sim = sim.reshape(1, 1, 64, 64)
    #         sim_1 = sim.squeeze(0).squeeze(0)
    #         plt.figure(figsize=(10, 10))
    #         # out = sim.softmax(dim=-1)
    #         plt.imshow(sim_1.cpu().detach().numpy())
    #         plt.savefig("feature_{}_sam.jpg".format(name),dpi=300)
    
    # # return test_image

    #         sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    #         sim = predictor.model.postprocess_masks(
    #                 sim,
    #                 input_size=predictor.input_size,
    #                 original_size=predictor.original_size).squeeze()

    #         topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=2)
    #         # topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    #         # topk_label = np.concatenate([last_label_i, topk_label_i], axis=0)

    #         topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    #         topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    #         sim = (sim - sim.mean()) / torch.std(sim)
    #         sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    #         attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    #         masks, scores, logits, _, upscaling = predictor.predict(
    #             point_coords=topk_xy,
    #             point_labels=topk_label,
    #             multimask_output=False,
    #             attn_sim=attn_sim,  # Target-guided Attention
    #             target_embedding=target_embedding_sam  # Target-semantic Prompting
    #         )
    #         from visual_feature import get_feature
    #         all_dict["fea_sam"] = upscaling
    #         get_feature(all_dict, "2", "2", name)

    #         best_idx = 0

    #         masks, scores, logits, _, _ = predictor.predict(
    #             point_coords=topk_xy,
    #             point_labels=topk_label,
    #             mask_input=logits[best_idx: best_idx + 1, :, :],
    #             multimask_output=True)
    #         best_idx = np.argmax(scores)

    #         # # Cascaded Post-refinement-2
    #         # y, x = np.nonzero(masks[best_idx])
    #         # if x.shape[0]!=0:
    #         #     x_min = x.min()
    #         #     x_max = x.max()
    #         #     y_min = y.min()
    #         #     y_max = y.max()
    #         #     input_box = np.array([x_min, y_min, x_max, y_max])
    #         #     masks, scores, logits, _ = predictor.predict(
    #         #         point_coords=topk_xy,
    #         #         point_labels=topk_label,
    #         #         box=input_box[None, :],
    #         #         mask_input=logits[best_idx: best_idx + 1, :, :],
    #         #         multimask_output=True)

    #         #     best_idx = np.argmax(scores)
    #         torch.cuda.synchronize()
    #         time_end = time.time()
        
    #         # Save masks
    #         sim = F.interpolate(sim, (448, 448), align_corners=True, mode='bilinear')
    #         sim = sim.squeeze(0).squeeze(0).cpu().numpy()
    #         plt.figure(figsize=(10, 10))
    #         plt.imshow(sim)
    #         # show_mask(masks[best_idx], plt.gca())
    #         show_points(topk_xy, topk_label, plt.gca())
    #         plt.title(f"Mask {best_idx}", fontsize=18)
    #         plt.axis('off')
    #         vis_mask_output_path = os.path.join("/newdata3/xsa/", 'vis_mask_111{}.jpg'.format(name))
    #         print(vis_mask_output_path)
    #         with open(vis_mask_output_path, 'wb') as outfile:
    #             plt.savefig(outfile, format='jpg')

    #         final_mask = masks[best_idx]
    #         mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    #         mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    #         mask_output_path = os.path.join("/newdata3/xsa/", "111" + '.png')
    #         cv2.imwrite(mask_output_path, mask_colors)

    #         final_mask = masks[best_idx]
    #         # print(np.unique(final_mask))
    #         final_masks.append(torch.tensor(final_mask))

    #     final_cls = torch.tensor(torch.stack(final_masks, dim=0),dtype=torch.float32)
    #     final_cls_avg = final_cls.mean(dim=0)
    #     final_masks_clss.append(final_cls_avg)
    
    # time_sum = time_end - time_start
    # global sum_time
    # sum_time += time_sum

    # final_masks_clss = torch.stack(final_masks_clss, dim=0)
    # scores = []

    print(torch.unique(final_masks_clss))

    for k in range(1, support_labels_onehot.shape[1]):
        # print(np.unique(final_masks_clss[k-1]))
        # save_mask = (final_masks_clss[k-1] > 0.5).cpu().numpy().astype(np.uint8)
        # save_colored_mask(save_mask, "panpan.png")
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

# @torch.no_grad()
# def inference_multi(model, image, label_onehot, support_images, support_labels_onehot, name, device):
#     n_labels = label_onehot.shape[0]
#     image, label_onehot = image.to(device), label_onehot.to(device)

#     image = F.interpolate(image.unsqueeze(0), (128, 128), align_corners=True, mode='bilinear').squeeze(0)
#     label_onehot = F.interpolate(label_onehot.unsqueeze(0), (128, 128), mode='nearest').squeeze(0)
#     support_labels_onehot = F.interpolate(support_labels_onehot, (128, 128), mode='nearest')
#     support_images = F.interpolate(support_images, (128, 128), align_corners=True, mode='bilinear')

#     support_size, _, h, w = support_images.shape

#     soft_pred_onehot = []
#     for k in range(n_labels):
#         support_labels = support_labels_onehot[:, k:k+1]
#         logits = model(
#             image[None],
#             support_images[None],
#             support_labels[None]
#         )[0]
#         soft_pred = torch.sigmoid(logits)
#         soft_pred_onehot.append(soft_pred)

#     soft_pred_onehot = torch.stack(soft_pred_onehot)
#     # hard_pred = F.softmax(soft_pred_onehot, dim=0)
#     # hard_pred_1 = torch.argmax(hard_pred, dim=0).squeeze(0)
#     hard_pred_1 = soft_pred_onehot.round().clip(0,1)

#     scores = []
#     for k in range(1, n_labels):
#         score = dice_score(hard_pred_1[k], label_onehot[k])
#         scores.append(score)
#         print('universeg:{}'.format(score))

#     # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
#     # for id in range(1, n_labels):
#     #     save_mask[0, 0, hard_pred_1 == id] = torch.tensor(id + 21, dtype=torch.uint8)
#     # save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
#     # save_mask = save_mask.squeeze(0).squeeze(0)
#     # save_colored_mask(np.array(save_mask), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/acdc_com/final_uni{}_{}_{}.png'.format(id, np.mean(scores), str(name).split("/")[-1])))


#     # backmask = (label_onehot[1:].sum(dim=0) == 0)
#     # label_onehot_save = torch.argmax(label_onehot, dim=0) + 21
#     # label_onehot_save[backmask] = 0

#     # label_onehot_save = F.interpolate(torch.tensor(label_onehot_save.unsqueeze(0).unsqueeze(0), dtype=torch.uint8), (448, 448), mode='nearest')
#     # label_onehot_save = label_onehot_save.squeeze(0).squeeze(0)
#     # save_colored_mask(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/acdc_com/label_{}_{}_{}.png'.format(id, np.mean(scores), str(name).split("/")[-1])))



#     return {'Image': image,
#             'Soft Prediction': soft_pred_onehot,
#             'Prediction': hard_pred_1,
#             'Ground Truth': label_onehot,
#             'score': np.mean(scores)}


#######  87多的模型
# acdc:  0.5818166872272821  0.6204415081738539
# wbc: 0.8707539583730737    0.8718685175125125
# scd: 0.5993716968590074    0.4872315970528617
# 

######  88多的模型
#  0.8707539583730735 0.8868995070530556
#  0.6760526477397062 0.6097241685807485
#  0.5993716968590075 0.5415119974021406
#  0.5591457569133169 0.6201066598439029
#  0.8884136320181203 0.8316973832651462
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
# # model_our = BaseModel(opt, build_model(opt)).cuda()
# # model_our.eval()


# # model_dict = model_our.state_dict()
# # check_decoder = torch.load(os.path.join('/data1/paintercoco/output_dir_scripple_sammul/', 'checkpoint-22.pth'))
# # for k, v in check_decoder.items():
# #     if k in model_dict.keys():
# #         model_dict[k] = v

# # model_our.load_state_dict(model_dict)
# # model_our.model.backbone = PeftModel.from_pretrained(model_our.model.backbone, "/data1/paintercoco/output_dir_scripple_sammul/22/")
# # print("load success")

# model_our_1 = MamICL(cfg=opt).cuda()
# model_our_1.eval()

# model_our_2 = MamICL(cfg=opt).cuda()
# model_our_2.eval()
# model_dict = model_our_2.state_dict()
# # # # # # # # # # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # # # # # # # # # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # # # # # # # # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # # # # # # # # # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # # # # # # # # # 
# # # # # # # # model_dict = model_our_1.state_dict()
# check_decoder = torch.load("/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_wfi/3200/checkpoint-3200_0.8531569199833696.pth")
# for k, v in check_decoder['model'].items():
#     if k in model_dict.keys():
#         model_dict[k] = v
# model_our_2.load_state_dict(model_dict)
# model_our_2.backbone = PeftModel.from_pretrained(model_our_2.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_wfi/3200/")


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


# model_seg = prepare_model_seggpt("/newdata3/xsa/seggpt_vit_large.pth",'seggpt_vit_large_patch16_input896x448', 'semantic').to('cuda')
# print('Model loaded.')


matcher = build_matcher_semantic_sam_oss(args)
# d_support = ACDCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = ACDCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # 0.59
# # 0.61
# d_support = SCDDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = SCDDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # 0.4709095757158112
# # 0.5242903106473097
# d_support = SPINEDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = SPINEDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # 0.63192027509891
# # 0.5683072578140393
# d_support = PanDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = PanDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # 0.9181284858381931
# # 0.8530985155415611
# d_support = WBCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = WBCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # 0.90
# # 0.80
# d_support = StareDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = StareDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# d_support = MonusegDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = MonusegDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

# d_support = MonusegDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = MonusegDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

d_support = HipXrayDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
d_test = HipXrayDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

# d_support = CervixDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = CervixDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

# # ######## 82.9 : acdc: 58.9   59.6  pandental没变  wbc到7.2 spine基本没变
# # ######  6200: spine:0.5566547957317339  

# # ####### 8800: acdc: 58.2  59.8  pandental没变  spine:0.5300699900820677
# # ########### 3200 scd: 47  50   pandental:0.849295971589358  spine搞了 0.5737823114581363   acdc： 0.597568082764105 0.5940644371611896  0.61295212381361  0.6842755436367238
# # ########### 3800 scd: 47                                    spine掉了 0.5422950657303583
# # ########### 4000 scd: 47  0.4600467432592702 acdc：0.6197818122940058  spine:0.5757235522191733  pandental; 0.8405215679948709  0.8748454341053072   stare 0.6615819640470441  0.6055563757629356
# # # #########  0.8565976445304151  0.853956173008545

# 0.259  17.3
# # #  universeg 64  scd: 0.608020624830973
# # #                scd: 0.16263674570235093
# 83.2 /// 0.5
backbone = get_dino_backbone(opt).cuda()

# 4983
cnt = 0
with torch.no_grad():
    for rep in range(repetation):
        n_support = 1
        support_images, support_labels, support_name = zip(*itertools.islice(d_support, n_support))
        print(support_name)
        # /support_images, support_labels, name = d_support

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
        # print(idxs)

        for i in tqdm(idxs):
            cnt +=1
            image, label, name = d_test[i]
            # name = "111"

            # vals_ours_2 = inference_multi(model_univer, image, label, support_images, support_labels, name, 'cuda')
            # vals_ours_2 = inference_seggpt(model_seg, image, label, support_images, support_labels, 'cuda')
            # vals_ours_2 = inference_multi_our(model_our_2, image, label, support_images, support_labels, 'cuda', name)
            # inference_multi_our(model_our_2, image, label, support_images, support_labels, 'cuda', name)
            vals_ours_2 = inference_multi_persam(backbone, image, label, support_images, support_labels, name)
            # vals_ours_2 = inference_multi_matcher(matcher, image, label, support_images, support_labels, name, 'cuda')
            # for k, v in vals_ours_1.items():
            #     results_univer[k].append(v)

            for k, v in vals_ours_2.items():
                results_ours_1[k].append(v)
            
    #         # for k, v in vals_ours_3.items():
    #         #     results_ours_2[k].append(v)
            
    #         # if vals_ours_2['score']> 0.45 and vals_ours_2['score'] > vals_ours_1['score'] and vals_ours_2['score'] > vals_ours_3['score']:
    #         #     save_colored_mask_1(np.array(vals_ours_2['save_mask']), "/newdata3/xsa/ICUSeg/mambamodel/eval/visual/ours_{}_{}_{}.png".format(name, support_name, vals_ours_2['score']))
    #         #     save_colored_mask_1(np.array(vals_ours_1['save_mask']), "/newdata3/xsa/ICUSeg/mambamodel/eval/visual/univer_{}.png".format(name))
    #         #     save_colored_mask_1(np.array(vals_ours_3['save_mask']), "/newdata3/xsa/ICUSeg/mambamodel/eval/visual/persam_{}.png".format(name))





        # scores_ours_1 = results_univer.pop('score')
        scores_ours_1 = results_ours_1.pop('score')
    #     # scores_ours_3 = results_ours_2.pop('score')

        avg_score_ours_1 = np.mean(scores_ours_1)
    #     # avg_score_ours_2 = np.mean(scores_ours_2)
    #     # avg_score_ours_3 = np.mean(scores_ours_3)

        total_ours_1.append(avg_score_ours_1)
    #     # total_ours_2.append(avg_score_ours_2)
    #     # total_ours_3.append(avg_score_ours_3)

    avg_ours_1 = np.mean(total_ours_1)
    std_ours_1 = np.std(total_ours_1)
    # print(total_ours_2)

    # avg_ours_2 = np.mean(total_ours_2)
    # std_ours_2 = np.std(total_ours_2)

    # avg_univer = np.mean(total_univer)
    # std_univer = np.std(total_univer)

    # print('univer avg dice score after 5 repetations:{}'.format(avg_univer))
    # print('univer std dice score after 5 repetations:{}'.format(std_univer))

    print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours_1))
    print('Our std dice score after 5 repetations:{}'.format(std_ours_1))

    # print('Our22 avg dice score after 5 repetations:{}'.format(avg_ours_1))
    # print('Our std dice score after 5 repetations:{}'.format(std_ours_1))
    print(sum_time / 100)


# 60.4  2.0