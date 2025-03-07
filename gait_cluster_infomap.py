# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import torch
import time
from datetime import timedelta
from torch.backends import cudnn
import argparse
from opengait.modeling import models
from opengait.utils import config_loader, params_count
import torch.distributed as dist

from opengait.utils import get_msg_mgr

from clustercontrast.utils.data import IterLoader
from clustercontrast.trainers import ClusterContrastTrainer

from label_refinement.utils_function import initialization

from label_refinement.utils_function import cluster_and_memory


def main():

    args = parser.parse_args()

    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True  # 告诉CuDNN在运行时根据硬件和输入数据的大小来选择最佳的卷积算法

    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))

    # load config file
    cfgs = config_loader(args.cfgs)
    training = (args.phase == 'train')

    initialization(cfgs, training)
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(args)

    cfgs['trainer_cfg']['restore_hint'] = args.checkpoint
    iters = args.iters if (args.iters > 0) else None

    # student branch model(Not participating in training)
    model_cfg = cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training=True)

    # teacher branch model
    ema_model = Model(cfgs, training=True)
    for param in ema_model.parameters():
        param.requires_grad = False

    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()

    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = ClusterContrastTrainer(model)

    if training:
        for epoch in range(args.epochs):

            pseudo_labels, memory, pseudo_labeled_dataset, refinement_dataset, part_score = cluster_and_memory(model, epoch, args, use_leg=False)

            args.eps  = args.eps * 0.97
            trainer.memory = memory
            trainer.ema_encoder = ema_model

            train_loader = IterLoader(model.train_loader, length=iters)

            train_loader.new_epoch()

            trainer.train(args, epoch, train_loader, optimizer, pseudo_labeled_dataset,
                          refinement_dataset,part_score, print_freq=args.print_freq, train_iters=len(train_loader))


            Model.run_test(model)

            if epoch %5==0 or epoch == args.epochs - 1:
                model.save_ckpt(epoch)

            lr_scheduler.step()

        end_time = time.monotonic()
        print('Total running time: ', timedelta(seconds=end_time - start_time))
    else:
        Model.run_test(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    # 需要调整的参数
    parser.add_argument('--eps', type=float, default=0.8 ,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")
    parser.add_argument('--k', type=int, default=2,
                        help="hyperparameter for outline")

    parser.add_argument('--refine_weight', type=float, default=0.4
                        , help="sigmoid function")
    parser.add_argument('--sig', type=int, default=30,
                        help="sigmoid function")
    parser.add_argument('--aug_weight', type=float, default=0.4,
                        help="sigmoid function")
    parser.add_argument('--center_sig', type=int, default=5,
                        help="sigmoid function")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=5)  # 将学习率衰减由20改为5
    parser.add_argument('--use_hard', type=bool, default=True)
    parser.add_argument('--use_refine_label', type=bool, default=True)
    parser.add_argument('--use_aug', type=bool, default=True)

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001,  # 修改：将学习率0.00035改成0.0001
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0005)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--cfgs', type=str, default="./configs/gaitgl/gaitgl_OU_CA.yaml",
                        help="temperature for scaling contrastive loss")

    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help="choose train or test phase")

    parser.add_argument('--checkpoint', type=str, default="./output/OUMVLP/GaitGL/GaitGL/checkpoints/GaitGL-210000.pt",
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')

    main()
