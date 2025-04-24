import torch.nn.functional as F
import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('/newdata3/xsa/UniverSeg-main')
sys.path.append('/newdata3/xsa/ICUSeg/mambamodel')

from universeg import universeg
import numpy as np
from example_Data.spine import SPINEDataset
import itertools
import math
import matplotlib.pyplot as plt
import einops as E
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from EICSeg import MamICL
import os
from PIL import Image
import imgviz
import imgviz
# from medpy.metric.binary import dc
from peft import PeftModel
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
timecnt = 0
timecnt_1 = 0

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
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/data1/output_dir_5/')
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
    # ref_masks[:, :n_labels, :, :] = label_onehot.unsqueeze(0).repeat(support_size, 1, 1, 1)[:, :, :, :]
    import time
    # torch.cuda.synchronize()
    time_start = time.time()

    logits = model(
        train_img,
        label_onehot,
        support_images,
        ref_masks, mode='test'
    )
    
    # torch.cuda.synchronize()
    # time_end = time.time()
    # time_sum = time_end - time_start
    # print("time_our:" , time_sum)
    # global timecnt
    # timecnt += time_sum

    soft_pred = torch.sigmoid(logits)
    # soft_pred = logits
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)  ###### (1, 10, 448, 448)
    hard_pred = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    print(torch.sum(hard_pred))

    save_mask = torch.zeros((h, w), dtype=torch.uint8)
    for id in range(n_labels):
        save_mask[hard_pred[id][0]] = torch.tensor(id + 14, dtype=torch.uint8)
        mask = hard_pred[id][0]
        mask[hard_pred[id][0]] = torch.tensor(id + 14, dtype=torch.uint8)
        mask = mask.cpu().numpy().astype(np.uint8)
        save_colored_mask(mask, os.path.join('{}_{}_.png'.format(id, id)))

    save_colored_mask(np.array(save_mask), os.path.join('final_{}.png'.format(id)))
    # backmask = label_onehot.sum(dim=0) == 0
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 14
    # label_onehot_save[backmask] = 0
    # save_colored_mask(np.array(label_onehot_save.cpu()), os.path.join('label_{}.png'.format(id)))

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

    import time
    # torch.cuda.synchronize()
    time_start = time.time()
    
    for _ in range(3):
        for k in range(n_labels):
            support_labels = support_labels_onehot[:,k:k+1]
            logits = model(
                image[None],
                support_images[None],
                support_labels[None]
            )[0]

            soft_pred = torch.sigmoid(logits)
            soft_pred_onehot.append(soft_pred)
    
    # torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    print("time_univer", time_sum)
    global timecnt_1
    timecnt_1 += time_sum

    soft_pred_onehot = torch.stack(soft_pred_onehot)
    hard_pred_1 = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)

    # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
    # for id in range(1, n_labels):
    #     save_mask[0, 0, hard_pred_1[id][0] == 1] = torch.tensor(id + 26, dtype=torch.uint8)
    #     mask = hard_pred_1[id][0]
    #     mask[hard_pred_1[id][0]] = torch.tensor(id + 13, dtype=torch.uint8)
    #     mask = mask.cpu().numpy().astype(np.uint8)
    #     save_colored_mask(mask, os.path.join('uni{}_{}_.png'.format(id, id)))

    # save_colored_mask(np.array(save_mask), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/final_uni{}_{}.png'.format(id, np.mean(scores))))
    # backmask = label_onehot.sum(dim=0) == 0
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 13
    # label_onehot_save[backmask] = 0
    # save_colored_mask(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/label_uni{}_{}_{}.png'.format(id, str(name).split("/")[-1], np.mean(scores))))
    scores = []
    for k in range(1, n_labels):
        score = dice_score(hard_pred_1[k], label_onehot[k])
        scores.append(score)
        print(score)
    
    # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
    # for id in range(1, n_labels):
    #     save_mask[0, 0, hard_pred_1[id][0] == 1] = torch.tensor(id + 24, dtype=torch.uint8)

    # save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    # save_mask = save_mask.squeeze(0).squeeze(0)
    # save_colored_mask(np.array(save_mask), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/final_uni{}_{}.png'.format(np.mean(scores), str(name).split("/")[-1])))

    # backmask = label_onehot[1:].sum(dim=0) == 0
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 24
    # label_onehot_save[backmask] = 0
    # save_colored_mask(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/spine_vis/label_uni{}_{}_{}.png'.format(id, np.mean(scores), str(name).split("/")[-1])))

    return {'Image': image,
            'Soft Prediction': soft_pred_onehot,
            'Prediction': hard_pred_1,
            'Ground Truth': label_onehot,
            'score': np.mean(scores)}



# args = get_args_parser()
# # model_univer = universeg(pretrained=True)
# # _ = model_univer.to('cpu')
# # model_univer.eval()

# opt = load_opt_from_config_files(args.conf_files)
# opt = init_distributed(opt)
# # # model_our = BaseModel(opt, build_model(opt)).cuda()
# # # model_our.eval()


# # # model_dict = model_our.state_dict()
# # # check_decoder = torch.load(os.path.join('/data1/paintercoco/output_dir_scripple_sammul/', 'checkpoint-22.pth'))
# # # for k, v in check_decoder.items():
# # #     if k in model_dict.keys():
# # #         model_dict[k] = v

# # # model_our.load_state_dict(model_dict)
# # # model_our.model.backbone = PeftModel.from_pretrained(model_our.model.backbone, "/data1/paintercoco/output_dir_scripple_sammul/22/")
# # # print("load success")

# model_our_1 = MamICL(cfg=opt).cuda()
# model_our_1.eval()

# model_our_2 = MamICL(cfg=opt).cuda()
# model_our_2.eval()

# # # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # # 
# model_dict = model_our_2.state_dict()
# check_decoder = torch.load("/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_onlydino/1400/checkpoint-1400_0.8553866446478365.pth")
# for k, v in check_decoder['model'].items():
#     if k in model_dict.keys():
#         model_dict[k] = v
# model_our_1.load_state_dict(model_dict)
# model_our_1.backbone = PeftModel.from_pretrained(model_our_1.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_onlydino/1400/")

# model_our_2.load_state_dict(model_dict)
# model_our_2.backbone = PeftModel.from_pretrained(model_our_2.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/7200/")
# print("load success")
# print("load success")

# # model_our = MamICL(cfg=opt).cuda()
# # model_our.eval()
# # # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # #                                                         #   output_device=args.local_rank, find_unused_parameters=True)

# # # # model_our = MamICL(cfg=opt).cuda()
# # # # model_our.eval()
# # model_dict = model_our.state_dict()
# # # # check_decoder = torch.load("/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/checkpoint-24400_0.49497267878995616.pth")
# # # # for k, v in check_decoder['model'].items():
# # # #     if k in model_dict.keys():
# # # #         model_dict[k] = v
# # # # model_our.load_state_dict(model_dict)
# # # # model_our.backbone = PeftModel.from_pretrained(model_our.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/")
# # check_decoder = torch.load("/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/7200/checkpoint-7200_0.8408008830264327.pth")
# # for k, v in check_decoder['model'].items():
# #     if k in model_dict.keys():
# #         model_dict[k] = v
# # model_our.load_state_dict(model_dict)
# # model_our.backbone = PeftModel.from_pretrained(model_our.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/7200/")

# # print("load success")


# repetation = 2
# total_univer = []
# total_ours = []
# total_ours_1 = []
# total_ours_2 = []

# # 8  1   16  1
# d_support = SPINEDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = SPINEDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# cnt = 0

# for rep in range(repetation):
#     n_support = 64
#     support_images, support_labels, name = zip(*itertools.islice(d_support, n_support))
#     support_images = torch.stack(support_images).to('cpu')
#     support_labels = torch.stack(support_labels).to('cpu')
    
#     n_viz = 16
#     n_predictions = 200
#     results_univer = defaultdict(list)
#     results_ours = defaultdict(list)
#     results_ours_1 = defaultdict(list)
#     results_ours_2 = defaultdict(list)
#     results_persam = defaultdict(list)

#     idxs = np.random.permutation(len(d_test))[:n_predictions]

#     for i in tqdm(idxs):
#         image, label, name = d_test[i]

#         # vals_univer = inference_multi(model_univer, image, label, support_images, support_labels, name, 'cpu')
#         vals_ours_1 = inference_multi_our(model_our_1, image, label, support_images, support_labels, 'cuda')
#         vals_ours_2 = inference_multi_our(model_our_1, image, label, support_images, support_labels, 'cuda')

#         # for k, v in vals_univer.items():
#         #     results_univer[k].append(v)

#         for k, v in vals_ours_1.items():
#             results_ours_1[k].append(v)
        
#         for k, v in vals_ours_2.items():
#             results_ours_2[k].append(v)


#     scores_ours_1 = results_ours_1.pop('score')
#     scores_ours_2 = results_ours_2.pop('score')
#     # scores_univer = results_univer.pop('score')

#     avg_score_ours_1 = np.mean(scores_ours_1)
#     avg_score_ours_2 = np.mean(scores_ours_2)
#     # avg_score_univer = np.mean(scores_univer)

#     total_ours_1.append(avg_score_ours_1)
#     total_ours_2.append(avg_score_ours_2)
#     # total_univer.append(avg_score_univer)

# avg_ours_1 = np.mean(total_ours_1)
# std_ours_1 = np.std(total_ours_1)

# avg_ours_2 = np.mean(total_ours_2)
# std_ours_2 = np.std(total_ours_2)

# # avg_univer = np.mean(total_univer)
# # std_univer = np.std(total_univer)

# # print('univer avg dice score after 5 repetations:{}'.format(avg_univer))
# # print('univer std dice score after 5 repetations:{}'.format(std_univer))

# print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours_1))
# print('Our std dice score after 5 repetations:{}'.format(std_ours_1))

# print('Our22 avg dice score after 5 repetations:{}'.format(avg_ours_2))
# print('Our std dice score after 5 repetations:{}'.format(std_ours_2))