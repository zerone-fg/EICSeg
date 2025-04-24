import torch.nn.functional as F
import torch
import os
from universeg import universeg
import numpy as np
from example_Data.monuseg import MonusegDataset
import itertools
import math
import matplotlib.pyplot as plt
import einops as E
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from example_Data.wbc import WBCDataset
from example_Data.acdc import ACDCDataset
from example_Data.pandental import PanDataset
from example_Data.SCD import SCDDataset
import os
from PIL import Image
import imgviz
import cv2
from peft import PeftModel, PeftConfig
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from EICSeg import MamICL
from vision_transformer import get_dino_backbone



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



def inference_multi_persam(model, image, label_onehot, support_images, support_labels_onehot, device):
    from PerSAM.per_segment_anything import sam_model_registry, SamPredictor
    sam_type, sam_ckpt = 'vit_b', '/newdata3/xsa/ICUSeg/ScribblePrompt_sam_v1_vit_b_res128.pt'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    sam.eval()

    test_image = Image.fromarray(np.uint8(image[0] * 255)).convert('RGB')
    test_image = np.array(test_image)
    

    predictor = SamPredictor(sam)
    predictor.set_image(test_image)
    test_feat_sam = predictor.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat_sam.shape
    test_feat_sam = test_feat_sam / test_feat_sam.norm(dim=0, keepdim=True)
    test_feat_sam = test_feat_sam.reshape(C, h * w)
    # predictor.set_image(test_image)


    images = torch.tensor(image).cuda()
    
    train_img = images[None].repeat(1, 3, 1, 1)
    features = model.get_intermediate_layers(train_img.float(), 3)  ##### 2, 1024, 32, 32

    final_masks_clss = []
    for l in range(1, support_labels_onehot.shape[1]):
        final_masks = []
        #对于每个类别，所有的ref_mask放在一起计算
        for k in range(support_images.shape[0]):
            ref_image_ori = Image.fromarray(np.uint8(support_images[k][0] * 255)).convert('RGB')
            ref_image_ori = np.array(ref_image_ori)

            ref_image = torch.tensor(ref_image_ori).cuda().unsqueeze(0)
            ref_image = ref_image.permute(0, 3, 1, 2)
            c, h, w = 3, 448, 448
            # ref_image = ref_image[None].repeat(1, 3, 1, 1)

            ref_mask_ori = Image.fromarray(np.uint8(support_labels_onehot[k][l]))
            ref_mask_ori_1 = np.array(Image.fromarray(np.uint8(support_labels_onehot[k][l]) * 255).convert('RGB'))
            ref_mask = torch.tensor(np.array(ref_mask_ori)).cuda().unsqueeze(0).unsqueeze(0)

            ref_mask_sam = predictor.set_image(ref_image_ori, ref_mask_ori_1)
            ref_feat_sam = predictor.features.squeeze().permute(1, 2, 0)
            ref_mask_sam = F.interpolate(ref_mask_sam, size=ref_feat_sam.shape[0: 2], mode="bilinear")
            ref_mask_sam = ref_mask_sam.squeeze()[0]

            # ref_mask = F.interpolate(ref_mask, size=ref_feat_sam.shape[0: 2], mode="nearest")
            # ref_mask = ref_mask.squeeze()[0]

            # Target feature extraction
            target_feat_sam = ref_feat_sam[ref_mask_sam > 0]
            target_embedding_sam = target_feat_sam.mean(0).unsqueeze(0)
            target_feat_sam = target_embedding_sam / target_embedding_sam.norm(dim=-1, keepdim=True)
            target_embedding_sam = target_embedding_sam.unsqueeze(0)

            

            ref_features = model.get_intermediate_layers(ref_image.float().reshape(-1, c, h, w), 3) #### 10, 1024, 32, 32

            fea_l = features['res2']
            rea_l = ref_features['res2']

            # fea_l = F.interpolate(fea_l, (16, 16), mode='bilinear', align_corners=True)
            # rea_l = F.interpolate(rea_l, (16, 16), mode='bilinear', align_corners=True)

            rea_l = rea_l.squeeze(0).permute(1, 2, 0)
            # fea_l = fea_l.reshape(1024, 32*32).mean(0).unsqueeze(0).transpose(-1, -2)
            # rea_l = rea_l.reshape(1024, 32*32).mean(0).unsqueeze(0)
            # sim = fea_l @ rea_l

            ref_mask = F.interpolate(ref_mask, size=rea_l.shape[0: 2], mode="nearest")
            ref_mask = ref_mask.squeeze()

            target_feat = rea_l[ref_mask > 0]
            target_embedding = target_feat.mean(0).unsqueeze(0)
            target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
            target_embedding = target_embedding.unsqueeze(0)

            # print('======> Start Testing')

            # predictor.set_image(test_image)
            # test_feat = predictor.features.squeeze()
            test_feat = fea_l.squeeze(0)
            C, h, w = test_feat.shape
            
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)


            # sim = target_feat_sam @ test_feat_sam
            sim = (target_feat @ test_feat)

            sim = sim.reshape(1, 1, 32, 32)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()

            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
            # topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            # topk_label = np.concatenate([last_label_i, topk_label_i], axis=0)

            topk_xy = np.concatenate([last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i], axis=0)

            sim = (sim - sim.mean()) / torch.std(sim)
            sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=False,
                attn_sim=attn_sim,  # Target-guided Attention
                target_embedding=target_embedding_sam  # Target-semantic Prompting
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
            
                # Save masks
                plt.figure(figsize=(10, 10))
                plt.imshow(test_image)
                show_mask(masks[best_idx], plt.gca())
                show_points(topk_xy, topk_label, plt.gca())
                plt.title(f"Mask {best_idx}", fontsize=18)
                plt.axis('off')
                vis_mask_output_path = os.path.join("/newdata3/xsa/", f'vis_mask_111.jpg')
                with open(vis_mask_output_path, 'wb') as outfile:
                    plt.savefig(outfile, format='jpg')

                final_mask = masks[best_idx]
                mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                mask_output_path = os.path.join("/newdata3/xsa/", "111" + '.png')
                cv2.imwrite(mask_output_path, mask_colors)

            final_mask = masks[best_idx]
            # print(np.unique(final_mask))
            final_masks.append(torch.tensor(final_mask))

        final_cls = torch.tensor(torch.stack(final_masks, dim=0),dtype=torch.float32)
        final_cls_avg = final_cls.mean(dim=0)
        final_masks_clss.append(final_cls_avg)

    final_masks_clss = torch.stack(final_masks_clss, dim=0)
    scores = []


    for k in range(1, support_labels_onehot.shape[1]):
        # print(np.unique(final_masks_clss[k-1]))
        save_mask = (final_masks_clss[k-1] > 0.5).cpu().numpy().astype(np.uint8)
        save_colored_mask(save_mask, "panpan.png")
        score = dice_score((final_masks_clss[k-1] > 0.5), label_onehot[k])
        scores.append(score)
        print('persam:{}'.format(score))

    return {'Image': image,
            'Soft Prediction': final_masks_clss[k-1],
            'Prediction': final_masks_clss[k-1],
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}


@torch.no_grad()
def inference_multi_our(model, image, label_onehot, support_images, support_labels_onehot, name, device, alpha=1):
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
        ref_masks, mode='test', alpha=alpha

    )

    soft_pred = torch.sigmoid(logits)
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)  ###### (1, 10, 448, 448)
    hard_pred = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)

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
    

@torch.no_grad()
def inference_multi(model, image, label_onehot, support_images, support_labels_onehot, name, device):
    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)

    image = F.interpolate(image.unsqueeze(0), (128, 128), align_corners=True, mode='bilinear').squeeze(0)
    label_onehot = F.interpolate(label_onehot.unsqueeze(0), (128, 128), mode='nearest').squeeze(0)
    support_labels_onehot = F.interpolate(support_labels_onehot, (128, 128), mode='nearest')
    support_images = F.interpolate(support_images, (128, 128), align_corners=True, mode='bilinear')

    support_size, _, h, w = support_images.shape

    soft_pred_onehot = []
    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        logits = model(
            image[None],
            # label_onehot,
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


args = get_args_parser()
model_univer = universeg(pretrained=True)
_ = model_univer.to('cpu')
model_univer.eval()

opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

backbone = get_dino_backbone(opt).cuda()

repetation = 5
total_univer = []
total_ours = []
total_ours_1 = []
total_ours_2 = []

d_support = MonusegDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
d_test = MonusegDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

for rep in range(repetation):
    n_support = 1
    support_images, support_labels, name = zip(*itertools.islice(d_support, n_support))
    support_images = torch.stack(support_images).to('cpu')
    support_labels = torch.stack(support_labels).to('cpu')

    n_viz = 16
    n_predictions = 10
    results_univer = defaultdict(list)
    results_ours = defaultdict(list)
    results_ours_1 = defaultdict(list)
    results_ours_2 = defaultdict(list)
    results_persam = defaultdict(list)

    idxs = np.random.permutation(len(d_test))[:n_predictions]

    for i in tqdm(idxs):
        image, label, name = d_test[i]
        vals_ours_1 = inference_multi_our(model_our_1, image, label, support_images, support_labels, name, 'cuda')

        for k, v in vals_ours_1.items():
            results_ours_1[k].append(v)


    scores_ours_1 = results_ours_1.pop('score')
    avg_score_ours_1 = np.mean(scores_ours_1)
    total_ours_1.append(avg_score_ours_1)

avg_ours_1 = np.mean(total_ours_1)
std_ours_1 = np.std(total_ours_1)

print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours_1))
print('Our std dice score after 5 repetations:{}'.format(std_ours_1))
