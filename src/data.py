from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
from param import args
from tok import Tokenizer
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import torch

DATA_ROOT = "dataset/"

def pil_loader(path):

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_saver(img, path):

    with open(path, 'wb') as f:
        img.save(f)

DEBUG_FAST_NUMBER = 1000

class DiffDataset:
    def __init__(self, ds_name='', split='train', task='ispeaker'):
        self.ds_name = ds_name
        self.split = split
        self.data = json.load(
            open(os.path.join(DATA_ROOT, self.ds_name, self.split + ".json"))
        )

        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "md_vocab.txt"))


class TorchDataset(Dataset):
    def __init__(self, dataset, pixels_f, task='speaker', max_length=80, 
                 img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name + "_" + dataset.split
        self.task = task
        self.tok = dataset.tok

        self.vis = pixels_f["image_features"]
        self.vis_box = pixels_f["spatial_features"]

        self.vis_concept = json.load(open(os.path.join(DATA_ROOT, self.dataset.ds_name, "image_concept.json")))
        self.imgid2id = json.load(open(os.path.join(DATA_ROOT, self.dataset.ds_name, "image_h5_id2idx.json")))
        self.object_num = json.load(open(os.path.join(DATA_ROOT, self.dataset.ds_name, "image_h5_len.json")))
        
        self.max_length_api = 9
        self.max_length_concept = 3
        self.max_length_hist = 33
        self.max_length_utte = 42

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, item):
        datum = self.dataset.data[item]
        uid = datum['uid']
        

        imgs = np.zeros((6,100,2048))
        imgs_mask = np.zeros((6,100))
        boxes = np.zeros((6,100,6))
        concept = np.ones((6, 100, self.max_length_concept), np.int64) * self.tok.pad_id
        concept_leng = np.ones((6,100), np.int64) * self.tok.pad_id
        for i, img in enumerate(datum['history_img']):
            img = img.strip()
            img_idx = self.imgid2id[img]  
            object_num = self.object_num[img]
            imgs_mask[i, :object_num] = 1
            imgs[i, :, :] = self.vis[img_idx]
            boxes[i, :, :] = self.vis_box[img_idx]

            concep_token = self.vis_concept[img_idx]
            concep_token = self.tok.encodes(concep_token)

            for o_num in range(len(concep_token)):
                leng_concept = len(concep_token[o_num])
                concept[i, o_num, :leng_concept] = concep_token[o_num]
                concept_leng[i, o_num] = leng_concept


        imgs = torch.from_numpy(imgs).float()
        boxes = torch.from_numpy(boxes).float()
        imgs_mask = torch.from_numpy(imgs_mask).float()
        img_leng = torch.tensor(len(datum['history_img']))

        concept = torch.from_numpy(concept)
        concept_leng = torch.from_numpy(concept_leng)

        api_gt = datum['api']
        api_gt = self.tok.encode(api_gt)
        length = len(api_gt)

        a = np.ones((self.max_length_api), np.int64) * self.tok.pad_id
        a[0] = self.tok.bos_id
        if length + 2 < self.max_length_api:       
            a[1: length+1] = api_gt
            a[length+1] = self.tok.eos_id
            length = 2 + length
        else:                                           
            a[1: -1] = api_gt[:self.max_length_api-2]
            a[self.max_length_api-1] = self.tok.eos_id      
            length = self.max_length_api

        api_gt = torch.from_numpy(a)
        api_leng = torch.tensor(length)



        hists = datum['history_utte']
        h = np.ones((self.max_length_hist, self.max_length_utte), np.int64) * self.tok.pad_id
        h_len = np.ones((self.max_length_hist), np.int64) * self.tok.pad_id
        for i, hist in enumerate(hists):
            utte = self.tok.encode(hist)
            leng_utte = len(utte)
            h[i, :leng_utte] = utte
            h_len[i] = leng_utte

        hists = torch.from_numpy(h)
        hists_len = torch.from_numpy(h_len)


        return uid, imgs, boxes, img_leng, imgs_mask, concept, concept_leng, api_gt, api_leng, hists, hists_len
       
