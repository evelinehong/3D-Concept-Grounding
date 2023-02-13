import sys
import os, os.path as osp
import configargparse
import torch
from torch.utils.data import DataLoader

import ndf.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf.training import losses, training, dataio
from ndf.utils import path_util

p = configargparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default=osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn'), help='root for logging')
p.add_argument('--experiment_name', type=str, default='try',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=200,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=5,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=5000,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=250,
               help='Training steps until save checkpoint')

p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
p.add_argument('--multiview_aug', action='store_true', help='multiview_augmentation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')

p.add_argument('--stage', type=int, default=0, help='curriculum learning')

p.add_argument('--vis', default=False, help='Visualization')

p.add_argument('--test', default=False, help='Inference')

p.add_argument('--vis_dir', default='visualization', help='Visualization')

opt = p.parse_args()

train_dataset = dataio.JointOccTrainDataset(128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, stage=opt.stage)
val_dataset = dataio.JointOccTrainDataset(128, phase='val', depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug, stage=opt.stage, vis=opt.vis, test=opt.test)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              drop_last=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                            drop_last=True, num_workers=4)

model = vnn_occupancy_network.VNNOccNet(latent_dim=256, return_features=True, stage=opt.stage).cuda()

if opt.checkpoint_path is not None:
    model.load_state_dict(torch.load(opt.checkpoint_path), strict=False)

# model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model_parallel = model

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)

all_losses = [losses.occupancy_net, losses.occupancy_net3, losses.occupancy_net2, losses.occupancy_net10, losses.occupancy_net9]

if (not opt.vis) and (not opt.test):
    training.train(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
                lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, all_losses=all_losses, iters_til_checkpoint=opt.iters_til_ckpt, 
                clip_grad=False, overwrite=False, stage=opt.stage)
elif opt.vis:
    training.vis(model=model_parallel, val_dataloader=val_dataloader, vis_dir=opt.vis_dir, stage=opt.stage)   
else:
    training.inference(model=model_parallel, val_dataloader=val_dataloader)

