import random
import numpy as np

import torch
import torchvision.transforms as transforms

from param import args
from speaker import Speaker
from data import DiffDataset, TorchDataset
import h5py

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    normalize
])

args.workers = 1   


# Loading Dataset
def get_tuple(ds_name, split, pixels_f, task='speaker', shuffle=True, drop_last=True):
    dataset = DiffDataset(ds_name, split, args.train)
    torch_ds = TorchDataset(dataset, pixels_f, task, max_length=args.max_input,
        img0_transform=img_transform, img1_transform=img_transform
    )
    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True,
        drop_last=drop_last)
    return dataset, torch_ds, loader

def main():

    pixels_f = h5py.File("dataset/multidial/mdial_h5.hdf5", 'r')
    train_tuple = get_tuple(args.dataset, 'md_train', pixels_f, shuffle=True, drop_last=True)
    valid_tuple = get_tuple(args.dataset, 'md_valid', pixels_f, shuffle=False, drop_last=False)
    speaker = Speaker(train_tuple[0]) 


    if args.load is not None:
        print("Load speaker from %s." % args.load)
        speaker.load(args.load)
        scores, result = speaker.evaluate(valid_tuple, split='md_valid')
        print("scores:", scores)
    if args.train == 'speaker':
        print("training..")
        speaker.train(train_tuple, valid_tuple, args.epochs)
    elif args.train == 'validspeaker':
        scores, result = speaker.evaluate(valid_tuple)
        print(scores)
    elif args.train == 'testspeaker':
        test_tuple = get_tuple(args.dataset, 'md_test', pixels_f, shuffle=False, drop_last=False)
        scores, result = speaker.evaluate(test_tuple, split='md_test')
        print("Test:")
        print("scores:", scores)

if __name__ == '__main__':
  main()
