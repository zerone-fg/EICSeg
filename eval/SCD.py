import torch.nn.functional as F
import torch
import os
import sys
from universeg import universeg
import numpy as np
from example_Data.SCD import SCDDataset
import itertools
import math
import matplotlib.pyplot as plt
import einops as E
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from peft import PeftModel, PeftConfig
import os
from PIL import Image
import imgviz
from EICSeg import MamICL
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

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

def define_colors_by_mean_sep(num_colors=133, channelsep=7):
    num_sep_per_channel = channelsep
    separation_per_channel = 256 // num_sep_per_channel

    color_dict = {}
    # R = G = B = 0
    # B += separation_per_channel  # offset for the first loop
    for location in range(num_colors):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_dict.values()

        color_dict[location] = (R, G, B)
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))
    return color_dict

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)

def save_colored_mask_1(mask, save_path):
    h, w = mask.shape
    save_mask = np.zeros((h, w, 3))
    for i in np.unique(mask):
        if i != 0:
            save_mask[mask == i] = color_map[i]
    cv2.imwrite(save_path, save_mask)


def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/data1/output_dir_simple_1/')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='/dataset/zhongqiaoyong/MedicalImages/CT/npy/CT_AbdomenCT-1K/imgs/CT_AbdomenCT-1K_Case_01008-034.npy')
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
def inference_multi_our(model, image, label_onehot, support_images, support_labels_onehot, device):
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

    ref_masks[:, :n_labels, :, :] = support_labels_onehot[:, :, :, :]

    logits = model(
        train_img,
        label_onehot,
        support_images,
        ref_masks, mode='test'
    )

    soft_pred = torch.sigmoid(logits)
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)  ###### (1, 10, 448, 448)
    hard_pred = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    print(hard_pred.sum())

    scores = []
    for k in range(n_labels):
        score = dice_score(hard_pred[k], label_onehot[k])
        scores.append(score)
        print(score)

    return {'Image': image,
            'Soft Prediction': soft_pred_onehot,
            'Prediction': hard_pred,
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}


def inference_multi_persam(model, image, label_onehot, support_images, support_labels_onehot, device):
    from xdecoder.body.ScribblePrompt.scribbleprompt import ScribblePromptSAM
    from xdecoder.body.PerSAM.per_segment_anything import sam_model_registry, SamPredictor
    import cv2
    from eval.DiceValExp.visual_feature import get_feature
    all_dict = {}
    # sam = ScribblePromptSAM()
    sam_type, sam_ckpt = 'vit_b', '/data1/ScribblePrompt_sam_v1_vit_b_res128.pt'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    sam.eval()

    test_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')
    test_image = np.array(test_image)

    final_masks_clss = []
    for l in range(1, support_labels_onehot.shape[1]):
        final_masks = []
        #对于每个类别，所有的ref_mask放在一起计算
        for k in range(support_images.shape[0]):
            ref_image = Image.fromarray(np.uint8(support_images[k][0] * 255)).convert('RGB')
            ref_image = np.array(ref_image)

            ref_mask = Image.fromarray(np.uint8(support_labels_onehot[k][l]* 255)).convert('RGB')
            ref_mask = np.array(ref_mask)

            predictor = SamPredictor(sam)
            ref_mask = predictor.set_image(ref_image, ref_mask)
            ref_feat = predictor.features.squeeze().permute(1, 2, 0)

            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
            ref_mask = ref_mask.squeeze()[0]

            target_feat = ref_feat[ref_mask > 0]
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
            target_embedding = target_embedding.unsqueeze(0)

            print('======> Start Testing')

            predictor.set_image(test_image)
            test_feat = predictor.features.squeeze()

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

            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
            topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

            sim = (sim - sim.mean()) / torch.std(sim)
            sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=False,
                attn_sim=attn_sim,  # Target-guided Attention
                target_embedding=target_embedding  # Target-semantic Prompting
            )
            best_idx = 0

            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            if x.shape[0]!=0:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks, scores, logits, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :],
                    multimask_output=True)

                best_idx = np.argmax(scores)

            final_mask = masks[best_idx]
            final_masks.append(torch.tensor(final_mask))

        final_cls = torch.tensor(torch.stack(final_masks, dim=0),dtype=torch.float32)
        final_cls_avg = final_cls.mean(dim=0)
        final_masks_clss.append(final_cls_avg)

    final_masks_clss = torch.stack(final_masks_clss, dim=0)
    scores = []
    for k in range(1, support_labels_onehot.shape[1]):
        score = dice_score(final_masks_clss[k-1], label_onehot[k])
        scores.append(score)
        print('persam:{}'.format(score))

    return {'Image': image,
            'Soft Prediction': final_masks_clss[k-1],
            'Prediction': final_masks_clss[k-1],
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}


@torch.no_grad()
def inference_multi(model, image, label_onehot, support_images, support_labels_onehot, name, device):
    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)

    image = F.interpolate(image.unsqueeze(0), (392, 392), align_corners=True, mode='bilinear').squeeze(0)
    label_onehot = F.interpolate(label_onehot.unsqueeze(0), (392, 392), mode='nearest').squeeze(0)
    support_labels_onehot = F.interpolate(support_labels_onehot, (392, 392), mode='nearest')
    support_images = F.interpolate(support_images, (392, 392), align_corners=True, mode='bilinear')

    support_size, _, h, w = support_images.shape

    soft_pred_onehot = []

    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        logits = model(
            image[None],
            support_images[None],
            support_labels[None]
        )[0]
        soft_pred = torch.sigmoid(logits)
        soft_pred_onehot.append(soft_pred)

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


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
def run_one_image_seggpt(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    # if model.seg_type == 'instance':
    #     seg_type = torch.ones([valid.shape[0], 1])
    # else:
    seg_type = torch.zeros([valid.shape[0], 1])

    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device),
                       valid.float().to(device), seg_type, feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1] // 2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output
    
def inference_multi_seggpt(model, image, label_onehot, support_images, support_labels_onehot, device):
    # 首先将label_onehot和suuport_label_hot处理为颜色值
    import json
    color_dict = define_colors_by_mean_sep(num_colors=133, channelsep=7)
    n_labels, h, w = label_onehot.shape  ### (5, 448, 448)
    image, label_onehot = image.to(device), label_onehot.to(device)
    support_size = support_labels_onehot.shape[0]

    train_img = image[None].repeat(1, 3, 1, 1)
    support_images = support_images.repeat(1, 3, 1, 1)
    ### image:(1, 448, 448) label:(5, 448, 448) support_imag: (56, 1, 448,448) support_label:(56, 5, 448, 448)

    new_seg_label = torch.zeros((1, 448, 448, 3), dtype=torch.long).cuda()
    new_seg_support = torch.zeros((support_labels_onehot.shape[0], 448, 448, 3), dtype=torch.long).cuda()

    new_seg_support = new_seg_support.view(-1, 3)
    new_seg_label = new_seg_label.view(-1, 3)

    label_onehot = label_onehot.view(n_labels, -1)
    support_labels_onehot = support_labels_onehot.permute(0, 2, 3, 1).reshape(-1, n_labels)

    for l in range(n_labels):
        color = color_dict[l]
        mask = label_onehot[l] == 1
        new_seg_label[mask] = torch.tensor(color).cuda()
        mask = support_labels_onehot[:, l] == 1
        new_seg_support[mask] = torch.tensor(color).cuda()

    new_seg_label = new_seg_label.reshape(1, 448, 448, 3)
    new_seg_support = new_seg_support.reshape(support_size, 448, 448, 3)

    image_batch, target_batch = [], []
    tgt = new_seg_label
    train_img = train_img[0].permute(1, 2, 0)
    support_images = support_images.permute(0, 2, 3, 1)
    tgt = tgt[0]

    for support_img, support_label in zip(support_images, new_seg_support):
        save_supp = Image.fromarray(support_label.cpu().numpy().astype(np.uint8))
        save_supp.save("supp.png")

        save_tgt = Image.fromarray(tgt.cpu().numpy().astype(np.uint8))
        save_tgt.save("tgt.png")

        tgt_1 = np.concatenate((support_label.cpu().numpy(), tgt.cpu().numpy()), axis=0)
        img_1 = np.concatenate((support_img.cpu().numpy(), train_img.cpu().numpy()), axis=0)

        img_1 = img_1 - imagenet_mean
        img_1 = img_1 / imagenet_std

        tgt_1 = tgt_1 - imagenet_mean
        tgt_1 = tgt_1 / imagenet_std

        image_batch.append(img_1)
        target_batch.append(tgt_1)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    torch.manual_seed(2)

    output = run_one_image_seggpt(img, tgt, model, device)

    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2),
        size=[448, 448],
        mode='nearest',
    ).permute(0, 2, 3, 1)[0].numpy()

    output = Image.fromarray(output.astype(np.uint8) * 10)
    output.save("seggpt.png")
    return output


def post_preprocess(mask, train_img):
    from xdecoder.body.ScribblePrompt.scribbleprompt import ScribblePromptSAM
    import cv2
    from eval.DiceValExp.visual_feature import get_feature
    all_dict = {}

    sam = ScribblePromptSAM()

    # predictor = SamPredictor(sam)
    mask = F.interpolate(mask, (1024, 1024), mode='bilinear', align_corners=True)
    mask_1 = F.interpolate(mask, (256, 256), mode='bilinear', align_corners=True)
    train_img = F.interpolate(train_img, (1024, 1024), mode='bilinear', align_corners=True)

    scribble = torch.cat(((mask > 0.5) == 1, (mask > 0.5) == 0), dim=1)
    # predictor.set_torch_image(train_img, (1024, 1024))

    topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(mask[0][0], topk=1000)
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    topk_xy = torch.tensor(topk_xy).unsqueeze(0).cuda()
    topk_label = torch.tensor(topk_label).unsqueeze(0).cuda()

    mask, img_features, low_res_logits = sam.predict(
        train_img,  # (B, 1, H, W)
        topk_xy,  # (B, n, 2)
        topk_label,  # (B, n)
        None,  # (B, 2, H, W)
        None,  # (B, n, 4)
        mask_1,  # (B, 1, 256, 256)
    )  # -> (B, 1, H, W), (B, 16, 256, 256), (B, 1, 256, 256)

    all_dict["final"] = img_features
    get_feature(all_dict, "1", "1")

    # # Cascaded Post-refinement-1
    mask_post = F.interpolate(mask, (256, 256), align_corners=True, mode='bilinear')
    mask, img_features, low_res_logits = sam.predict(
        train_img,  # (B, 1, H, W)
        topk_xy,  # (B, n, 2)
        topk_label,  # (B, n)
        None,  # (B, 2, H, W)
        None,  # (B, n, 4)
        mask_post,  # (B, 1, 256, 256)
    )  # -> (B, 1, H, W), (B, 16, 256, 256), (B, 1, 256, 256)
    refine_mask = F.interpolate(mask, (448, 448), mode='bilinear', align_corners=True)
    refine_mask = refine_mask > 0.5

    mask = mask.cpu().numpy() > 0.5
    topk_xy, topk_label = topk_xy[0].cpu().numpy(), topk_label[0].cpu().numpy()

    test_image = np.uint8(train_img[0].permute(1, 2, 0).cpu().numpy() * 255)

    plt.figure(figsize=(10, 10))
    plt.imshow(test_image)
    show_mask(mask, plt.gca())
    show_points(topk_xy, topk_label, plt.gca())
    plt.title(f"Mask {0}", fontsize=18)
    plt.axis('off')
    vis_mask_output_path = os.path.join("./", f'vis_mask_{0}.jpg')
    with open(vis_mask_output_path, 'wb') as outfile:
        plt.savefig(outfile, format='jpg')

    final_mask = mask[0][0]
    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
    mask_output_path = os.path.join("./", str(0) + '.png')
    cv2.imwrite(mask_output_path, mask_colors)
    return refine_mask
def prepare_model_seggpt(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    import xdecoder.models_seggpt as models_seggpt

    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
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


args = get_args_parser()
model_univer = universeg(pretrained=True)
_ = model_univer.to('cpu')
model_univer.eval()

opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

model_our= MamICL(cfg=opt).cuda()
model_our.eval()


model_dict = model_our.state_dict()
check_decoder = torch.load("checkpoint-7200.pth")
for k, v in check_decoder['model'].items():
    if k in model_dict.keys():
        model_dict[k] = v
model_our.load_state_dict(model_dict)
model_our.backbone = PeftModel.from_pretrained(model_our.backbone, "/newdata3/7200/")


repetation = 200
total_ours = []


d_support = SCDDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
d_test = SCDDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

cnt = 0
for rep in range(repetation):
    n_support = 64
    support_images, support_labels, name = zip(*itertools.islice(d_support, n_support))
    support_images = torch.stack(support_images).to('cpu')
    support_labels = torch.stack(support_labels).to('cpu')

    n_viz = 16
    n_predictions = 10
    results_ours = defaultdict(list)

    idxs = np.random.permutation(len(d_test))[:n_predictions]

    for i in tqdm(idxs):
        cnt += 1
        image, label, name = d_test[i]
        vals_ours_1 = inference_multi_our(model_our, image, label, support_images, support_labels, 'cuda')

        for k, v in vals_ours_1.items():
            results_ours[k].append(v)

    scores_ours = results_ours.pop('score')

    avg_score_ours = np.mean(scores_ours)

    total_ours.append(avg_score_ours)

avg_ours = np.mean(total_ours)
std_ours = np.std(total_ours)

print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours))
print('Our std dice score after 5 repetations:{}'.format(std_ours))
