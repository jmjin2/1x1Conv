from pathlib import Path
from torch.utils import data as data
from utils.util import imfrombytes, img2tensor, FileClient
import random
import torch
import os

class MIVRecurrentDataset(data.Dataset):
    def __init__(self):
        super(MIVRecurrentDataset, self).__init__()
        self.gt_root, self.lq_root = Path("datasets/Train/GT"), Path("datasets/Train/x4")
        self.num_frame = 15

        self.keys = []
        with open('data/meta_info_GT.txt', 'r') as fin:
            for line in fin:
                folder, frame_num, extention, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}/{frame_num}/{extention}' for i in range(int(frame_num))])
        
        # file client (io backend)
        self.file_client = None
        self.interval_list = [1]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient('disk')

        key = self.keys[index]
        clip_name, texture, view, frame_name, frame_num, extention = key.split('/')
        view_list = os.listdir(self.gt_root / clip_name / texture)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        frame_num = int(frame_num)
        if start_frame_idx > frame_num - self.num_frame * interval:
            start_frame_idx = random.randint(0, frame_num - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # get the neighboring LQ and GT frames
        img_lq1s = []
        img_lq2s = []
        img_lq3s = []
        img_gts = []
     
        for neighbor in neighbor_list:
            if extention == 'jpg':
                img_gt_path = self.gt_root / clip_name / texture/ view / f'{neighbor:08d}.jpg'
            elif extention == 'png':
                img_gt_path = self.gt_root / clip_name / texture/ view / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            if extention == 'jpg':
                img_lq1_path = self.lq_root / clip_name / texture/ view_list[0] / f'{neighbor:08d}.jpg'
            elif extention == 'png':
                img_lq1_path = self.lq_root / clip_name / texture/ view_list[0] / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq1_path, 'lq1')
            img_lq1 = imfrombytes(img_bytes, float32=True)
            img_lq1s.append(img_lq1)

            if extention == 'jpg':
                img_lq2_path = self.lq_root / clip_name / texture/ view / f'{neighbor:08d}.jpg'
            elif extention == 'png':
                img_lq2_path = self.lq_root / clip_name / texture/ view / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq2_path, 'lq2')
            img_lq2 = imfrombytes(img_bytes, float32=True)
            img_lq2s.append(img_lq2)

            if extention == 'jpg':
                img_lq3_path = self.lq_root / clip_name / texture/ view_list[-1] / f'{neighbor:08d}.jpg'
            elif extention == 'png':
                img_lq3_path = self.lq_root / clip_name / texture/ view_list[-1] / f'{neighbor:08d}.png'
            img_bytes = self.file_client.get(img_lq3_path, 'lq3')
            img_lq3 = imfrombytes(img_bytes, float32=True)
            img_lq3s.append(img_lq3)

        img_gts, img_lq1s, img_lq2s, img_lq3s = paired_random_crop(img_gts, img_lq1s, img_lq2s, img_lq3s, 256, 4, img_gt_path)

        img_gts = torch.stack(img2tensor(img_gts), dim=0)
        img_lq1s = torch.stack(img2tensor(img_lq1s), dim=0)
        img_lq2s = torch.stack(img2tensor(img_lq2s), dim=0)
        img_lq3s = torch.stack(img2tensor(img_lq3s), dim=0)

        return {'lq1': img_lq1s, 'lq2': img_lq2s, 'lq3': img_lq3s, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

def paired_random_crop(img_gts, img_lq1s, img_lq2s, img_lq3s, gt_patch_size, scale, gt_path=None):
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lq2s[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lq2s[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lq1s = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lq1s]
        img_lq2s = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lq2s]
        img_lq3s = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lq3s]
    else:
        img_lq1s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq1s]
        img_lq2s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq2s]
        img_lq3s = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lq3s]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    return img_gts, img_lq1s, img_lq2s, img_lq3s