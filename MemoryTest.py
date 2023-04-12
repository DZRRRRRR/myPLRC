import os
import psutil
import torch
from PLRC.datasets.DAPreDataset import DAPre
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from PLRC.datasets.DAPreDataset import DAPre
from PLRC.models.builder import PLRC

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")


parser.add_argument("--data",default="D:/chu/workspace/dataset/OfficeHomeDataset", help="path to dataset")
parser.add_argument("--dataset",default="OfficeHome")
parser.add_argument("--testEnv",default=0)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:10001",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="gloo", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=3407, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=512,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)


# plrc configs:
parser.add_argument(
    "--ts_ratio",
    default=0.3,
    type=float,
    help="balancing factor for self-distillation loss",
)
parser.add_argument(
    "--cl_ratio",
    default=0.3,
    type=float,
    help="balancing factor for point-level contrastive loss",
)
parser.add_argument(
    "--im_ratio",
    default=0.7,
    type=float,
    help="balancing factor for image-level contrastive loss",
)

def log_device_usage(count, use_cuda):
    mem_Mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cuda_mem_Mb = torch.cuda.memory_allocated(0) / 1024 ** 2 if use_cuda else 0
    print(f"iter {count}, mem: {int(mem_Mb)}Mb, gpu mem:{int(cuda_mem_Mb)}Mb")


if __name__ == "__main__":
    train_dataset = DAPre(0,None,"D:/chu/workspace/dataset/OfficeHomeDataset")
    train_sampler = None
    args = parser.parse_args()
    args.gpu = 0
    use_cuda = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=(train_sampler is None),
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    
    
    model = PLRC(args)
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    torch.backends.cudnn.benchmark = True
    for cur_iter, (
        inputs,
        ids,
        cls_labels,
        masks,
        obj,
        inputs_2,
        coord_multiv,
        coord_multiv_2,
    ) in enumerate(train_loader):
        # print("*************************************************************")
        # print("input0:",inputs[0].shape,"inputs1.shape:",inputs[1].shape)
        # print("ids:",ids.shape)
        # print("cls_labels:",cls_labels.shape)
        # print("masks0:",masks[0].shape,"masks1.shape:",masks[1].shape)
        # print("obj:",obj.shape)
        # print("inputs_2.0:",inputs_2[0].shape,"inputs_2.1:",inputs_2[1].shape)
        # print("coord_multiv0:",coord_multiv[0].shape,"coord_multiv1:",coord_multiv[1].shape)
        # print("coord_multiv_20:",coord_multiv[0].shape,"coord_multiv_21:",coord_multiv[1].shape)
        
        
        if(ids[0]!=3489):
            continue
        
        
        while(True):
            cur_iter = cur_iter+1
            if args.gpu is not None:

                inputs_cuda = []
                masks_cuda = []
                inputs_2_cuda = []
                for i in range(len(inputs)):
                    ii = inputs[i].float().cuda(non_blocking=True)
                    jj = masks[i].float().cuda(non_blocking=True)
                    kk = inputs_2[i].float().cuda(non_blocking=True)
                    # hack to squeeze the extra dimension, need to think about loss balancing if there are multiple
                    if len(ii.shape) == 5:
                        _, dm, dc, dh, dw = ii.shape
                        _, dm, dc, dh, dw = jj.shape
                        ii = torch.reshape(ii, (-1, dc, dh, dw))
                        jj = torch.reshape(jj, (-1, dc, dh, dw))
                        kk = torch.reshape(jj, (-1, dc, dh, dw))
                    inputs_cuda.append(ii)
                    masks_cuda.append(jj)
                    inputs_2_cuda.append(kk)
                # del inputs, masks, inputs_2
                inputs = inputs_cuda
                masks = masks_cuda
                inputs_2 = inputs_2_cuda
                ids = ids.cuda(non_blocking=True)
                
            loss_point, loss_moco = model(
                inputs, ids, 0, masks, obj, inputs_2, coord_multiv, coord_multiv_2
            )
            loss_vec_point, to_vis_point = loss_point

            loss_vec_moco, to_vis_moco = loss_moco

            loss = loss_vec_point.sum() + loss_vec_moco.sum() * args.im_ratio
                
            optimizer.zero_grad()
            loss.backward() # backward 降显存
            optimizer.step()
        
        
        # log_device_usage(cur_iter,use_cuda)
        # torch.cuda.empty_cache()
        # log_device_usage(cur_iter,use_cuda)
