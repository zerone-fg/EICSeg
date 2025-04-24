import torch.nn.functional as F
import torch
import os
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('/newdata3/xsa/UniverSeg-main')
sys.path.append('/newdata3/xsa/ICUSeg/mambamodel')

from universeg import universeg
import numpy as np
from example_Data.wbc import WBCDataset
from example_Data.stare import StareDataset
from example_Data.pandental import PanDataset
from example_Data.acdc import ACDCDataset
from example_Data.hipxray import HipXrayDataset
from example_Data.monuseg import MonusegDataset
from example_Data.spine import SPINEDataset
from example_Data.SCD import SCDDataset
import itertools
import math
import matplotlib.pyplot as plt
import einops as E
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
# from EICSeg import MamICL
# from MamICL_invo import MamICL
# from util.distributed import init_distributed
# from util.arguments import load_opt_from_config_files
# from xdecoder.BaseModel import BaseModel
# from xdecoder import build_model
from EICSeg_swinunet import MamICL
from PIL import Image
import os
import imgviz
# from medpy.metric.binary import dc
import cv2
from peft import PeftModel, PeftConfig
# from utils.distributed import init_distributed
# from utils.arguments import load_opt_from_config_files
# from Painter_main.SegGPT.seggpt_engine import run_one_image
# from Painter_main.SegGPT.SegGPT_inference import models_seggpt
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from config import get_config

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

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


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)


def save_colored_mask_1(mask, save_path):
    h, w = mask.shape
    save_mask = np.zeros((h, w, 3))
    print(np.unique(mask))
    for i in np.unique(mask):
        if i != 0:
            save_mask[mask == i] = color_map[i]
    cv2.imwrite(save_path, save_mask)


def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)
    parser.add_argument('--batch_size', default=80
    , type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
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
    ##########新增加的#####################################################
    parser.add_argument('--cfg', default="/newdata3/xsa/ICUSeg/mambamodel/configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
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

# def inference_multi_seggpt(model, image, label_onehot, support_images, support_labels_onehot, device):
#     label_onehot = label_onehot[1:]
#     support_labels_onehot = support_labels_onehot[:, 1:]

#     n_labels = label_onehot.shape[0]
#     image, label_onehot = image.to(device), label_onehot.to(device)

#     support_size, _, h, w = support_images.shape
#     image = (image - image.min()) / (image.max() - image.min())
#     support_images = (support_images - support_images.min()) / (support_images.max() - support_images.min())
    

#     train_img = image[None].repeat(1, 3, 1, 1)
#     support_images = support_images.repeat(1, 3, 1, 1)


#     arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'
#     model = getattr(models_seggpt, arch)()
#     model.seg_type = seg_type
#     checkpoint = torch.load(chkpt_dir, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     model.eval()

#     print('Model loaded.')

#     ####### img为query image和prompt images拼在一起
#     ####### label为query label和prompt labels拼在一起
#     train_img = image[None].repeat(1, 3, 1, 1)
#     support_images = support_images.repeat(1, 3, 1, 1)

#     img = torch.concatenate((train_img, support_images), dim=0)
#     tgt = 


#     output = run_one_image(img, tgt, model, device)
#     output = F.interpolate(
#         output[None, ...].permute(0, 3, 1, 2), 
#         size=[size[1], size[0]], 
#         mode='nearest',
#     ).permute(0, 2, 3, 1)[0].numpy()
#     output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))



#################################### ICUSeg测试代码 #####################################################
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
    # soft_pred = logits
    soft_pred_onehot = soft_pred[:, :n_labels, :, :].transpose(0, 1)  ###### (1, 10, 448, 448)
    hard_pred = torch.tensor(soft_pred_onehot > 0.5, dtype=torch.uint8)
    print(torch.sum(hard_pred))

    # ############################### 给预测结果进行上色，并保存 ###################################
    # save_mask = torch.zeros((h, w), dtype=torch.uint8)
    # for id in range(n_labels):
    #     save_mask[hard_pred[id][0]] = torch.tensor(id + 6, dtype=torch.uint8)
    #     mask = hard_pred[id][0]
    #     mask[hard_pred[id][0] == 1] = torch.tensor(id + 6, dtype=torch.uint8)
    #     mask = mask.cpu().numpy().astype(np.uint8)
    #     save_colored_mask(mask, os.path.join('{}_{}_.png'.format(id, id)))

    # save_colored_mask_1(np.array(save_mask), os.path.join('final_{}.png'.format(id)))

    # ############################### 对GT进行相同的着色 #########################################
    # backmask = label_onehot.sum(dim=0) == 0
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 6
    # label_onehot_save[backmask] = 0
    # save_colored_mask_1(np.array(label_onehot_save.cpu()), os.path.join('label_{}.png'.format(id)))


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

#################################### Universeg测试代码 #####################################################
@torch.no_grad()
def inference_multi(model, image, label_onehot, support_images, support_labels_onehot, name, device):
    n_labels = label_onehot.shape[0]
    image, label_onehot = image.to(device), label_onehot.to(device)  ##### (1, 448, 448)  (3, 448, 448)

    image = F.interpolate(image.unsqueeze(0), (128, 128), align_corners=True, mode='bilinear').squeeze(0)
    label_onehot = F.interpolate(label_onehot.unsqueeze(0), (128, 128), mode='nearest').squeeze(0)
    support_labels_onehot = F.interpolate(support_labels_onehot, (128, 128), mode='nearest')
    support_images = F.interpolate(support_images, (128, 128), align_corners=True, mode='bilinear')
    support_size, _, h, w = support_images.shape

    support_images = support_images.to(device)

    soft_pred_onehot = []
    for k in range(n_labels):
        support_labels = support_labels_onehot[:,k:k+1]
        support_labels = support_labels.to(device)

        logits = model(
            image[None],
            label_onehot,
            support_images[None],
            support_labels[None]
        )[0]
        soft_pred = torch.sigmoid(logits)
        soft_pred_onehot.append(soft_pred)

    soft_pred_onehot = torch.stack(soft_pred_onehot)  ##### (3, 1, 448, 448)
    hard_pred = F.softmax(10 * soft_pred_onehot,dim=0)  #### (3, 1, 448, 448)
    hard_pred_1 = torch.argmax(hard_pred, dim=0).squeeze(0)

    scores = []

    for k in range(1, n_labels):
        score = dice_score(hard_pred_1 == k, label_onehot[k])
        scores.append(score)
        print(score)


    # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)
    # for id in range(1, n_labels):
    #     save_mask[0, 0, hard_pred_1 == id] = torch.tensor(id + 4, dtype=torch.uint8)
    # save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    # save_mask = save_mask.squeeze(0).squeeze(0)
    # save_colored_mask_1(np.array(save_mask), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/wbc_com/final_uni{}_{}_{}.png'.format(id, np.mean(scores), str(name).split("/")[-1])))
    

    # backmask = (label_onehot[1:].sum(dim=0) == 0)
    # label_onehot_save = torch.argmax(label_onehot, dim=0) + 4
    # label_onehot_save[backmask] = 0

    # label_onehot_save = F.interpolate(torch.tensor(label_onehot_save.unsqueeze(0).unsqueeze(0), dtype=torch.uint8), (448, 448), mode='nearest')
    # label_onehot_save = label_onehot_save.squeeze(0).squeeze(0)
    # save_colored_mask_1(np.array(label_onehot_save.cpu()), os.path.join('/newdata3/xsa/ICUSeg/mambamodel/eval/wbc_com/label_{}_{}_{}.png'.format(id, np.mean(scores), str(name).split("/")[-1])))
    


    return {'Image': image,  ## float32
            'Soft Prediction': soft_pred_onehot,  ### float32
            'Prediction': hard_pred_1,  #### float32
            'Ground Truth': label_onehot,  ### float32
            'score': np.mean(scores)}





# args = get_args_parser()
# model_univer = universeg(pretrained=True)
# _ = model_univer.to('cpu')
# model_univer.eval()

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

# # model_our_1 = MamICL(cfg=opt).cuda()
# config = get_config(args)
# model_our_1 = MamICL(cfg=opt, config=config).cuda()
# model_our_1.eval()

# # model_our_2 = MamICL(cfg=opt).cuda()
# # model_our_2.eval()

# # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # model_our = torch.nn.parallel.DistributedDataParallel(model_our, device_ids=[args.local_rank],
# # #                                                           output_device=args.local_rank, find_unused_parameters=True)
# # # 

# # model_our_2.backbone = PeftModel.from_pretrained(model_our_2.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_224_ctm_sam_qkv_clsfusion_nochannel_wsam_1/7200/")
# # print("load success")
# # print("load success")



# model_dict = model_our_1.state_dict()
# check_decoder = torch.load("/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_swin/1600/checkpoint-1600_0.2655645841572927.pth")
# for k, v in check_decoder['model'].items():
#     if k in model_dict.keys():
#         model_dict[k] = v
# model_our_1.load_state_dict(model_dict)

# # model_our_1.backbone = PeftModel.from_pretrained(model_our_1.backbone, "/newdata3/xsa/ICUSeg/mambamodel/output_dir_dino_spsize448_onlydino/16600/")
# repetation = 5
# total_univer = []
# total_ours = []
# total_ours_1 = []
# total_ours_2 = []


# # d_support = WBCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = WBCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = StareDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = StareDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = PanDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = PanDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = ACDCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = ACDCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = ACDCDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = ACDCDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = HipXrayDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = HipXrayDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = MonusegDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = MonusegDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# # d_support = SPINEDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# # d_test = SPINEDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))
# d_support = SCDDataset('JTSC', split='support', label=None, size=(args.input_size, args.input_size))
# d_test = SCDDataset('JTSC', split='test', label=None, size=(args.input_size, args.input_size))

# support_txt = open("wbc_64_train.txt", "a")
# test_txt = open("wbc_64_test.txt", "a")

# cnt = 0

# for rep in range(repetation):
#     n_support = 64
#     support_images, support_labels, _ = zip(*itertools.islice(d_support, n_support))
#     # for file in name:
#     #     support_txt.write(str(file) + "\n")

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
#         cnt += 1
#         image, label, _ = d_test[i]

#         # vals_univer = inference_multi(model_univer, image, label, support_images, support_labels, cnt, 'cpu')
#         vals_ours_1 = inference_multi_our(model_our_1, image, label, support_images, support_labels, 'cuda', alpha=1/2)
#         # vals_ours_2 = inference_multi_our(model_our_1, image, label, support_images, support_labels, 'cuda', alpha=1/16)

#         # for k, v in vals_univer.items():
#         #     results_univer[k].append(v)

#         for k, v in vals_ours_1.items():
#             results_ours_1[k].append(v)
        
#         # for k, v in vals_ours_2.items():
#         #     results_ours_2[k].append(v)


#     scores_ours_1 = results_ours_1.pop('score')
#     # scores_ours_2 = results_ours_2.pop('score')
#     # scores_univer = results_univer.pop('score')

#     avg_score_ours_1 = np.mean(scores_ours_1)
#     # avg_score_ours_2 = np.mean(scores_ours_2)
#     # avg_score_univer = np.mean(scores_univer)

#     total_ours_1.append(avg_score_ours_1)
#     # total_ours_2.append(avg_score_ours_2)
#     # total_univer.append(avg_score_univer)

# avg_ours_1 = np.mean(total_ours_1)
# std_ours_1 = np.std(total_ours_1)

# # avg_ours_2 = np.mean(total_ours_2)
# # std_ours_2 = np.std(total_ours_2)

# # avg_univer = np.mean(total_univer)
# # std_univer = np.std(total_univer)

# # print('univer avg dice score after 5 repetations:{}'.format(avg_univer))
# # print('univer std dice score after 5 repetations:{}'.format(std_univer))

# print('Our11 avg dice score after 5 repetations:{}'.format(avg_ours_1))
# print('Our std dice score after 5 repetations:{}'.format(std_ours_1))

# # print('Our22 avg dice score after 5 repetations:{}'.format(avg_ours_2))
# # print('Our std dice score after 5 repetations:{}'.format(std_ours_2))