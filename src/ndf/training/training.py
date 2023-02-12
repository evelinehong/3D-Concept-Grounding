'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from collections import defaultdict
import torch.distributed as dist
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
import random

import ndf.training.util as util

color_list = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, all_losses,
          iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, 
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None, stage=0):
                             
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if stage not in [0,1,2]:
        for name, child in model.named_children():
            if "encoder" in name or "decoder" in name:
                util.dfs_freeze(child)
    #     param.requires_grad = False
    
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    loss_fn = all_losses[0]

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []

        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))

                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)

                losses = all_losses[0](model_output, gt)

                if stage != 0:
                    losses.update(all_losses[1](model_output['color'], model_input['coord_color'], gt))

                if not stage in [0, 1]:

                    if model_input['question'][0][0][0] in [0,1] and model_input['question'][0][1][0] == -1:
                        losses.update(all_losses[2](model_output['ans'][0], model_input['answer']))
                    if model_input['question'][0][0][0] in [0,1] and model_input['question'][0][1][0] != -1:
                        losses.update(all_losses[3](model_output['ans'][0], model_input['answer']))

                if stage == 4 and model_input['question'][0][1][0] == 4:
                    scores = model_output['ans'][1]

                    losses.update(all_losses[4](model_output['ans'][0], model_input['answer'], scores))

                train_loss = 0. 

                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and total_steps != 0 and rank == 0:
                    # print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    total_acc0 = 0.0
                    total_acc1 = 0.0
                    total_acc2 = 0.0

                    total_sample = 0
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)

                            m = 0; n = 0;
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)

                                model_output = model(model_input)
                                
                                val_loss = all_losses[0](model_output, gt, val=True)
                                
                                if not stage == 0:
                                    val_loss.update(all_losses[1](model_output['color'], model_input['coord_color'], gt))

                                if not stage in [0, 1]:
                                    if model_input['question'][0][0][0] in [0,1] and model_input['question'][0][1][0] == -1:
                                        val_loss.update(all_losses[2](model_output['ans'][0], model_input['answer']))
                                    if model_input['question'][0][0][0] in [0,1] and model_input['question'][0][1][0] != -1:
                                        val_loss.update(all_losses[3](model_output['ans'][0], model_input['answer']))

                                if stage == 4 and model_input['question'][0][1][0] == 4:
                                    scores = model_output['ans'][1]

                                    val_loss.update(all_losses[4](model_output['ans'][0], model_input['answer'], scores))

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)
                            print ('val_' + loss_name)
                            print (single_loss)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        return model, optimizers

def vis(model, val_dataloader=None, vis_dir=None, stage=0):

    if not vis_dir in os.listdir("./"):
        os.mkdir(vis_dir)
    if not str(stage) in os.listdir(vis_dir):
        os.mkdir(os.path.join(vis_dir, str(stage)))
    
    cur_vis_dir = os.path.join(vis_dir, str(stage))

    if val_dataloader is not None:
        print("Running validation set...")
        with torch.no_grad():
            model.eval()

            m = 0; n = 0;
            for val_i, (model_input, gt) in enumerate(val_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                model_output = model(model_input)

                for j in range(model_input['coords'].shape[0]):
                        idxes = torch.where(model_output['occ'][j] > 0.5)[0].cpu()

                        points = torch.index_select(model_input['coords'][j].cpu(), 0, idxes).numpy()

                        if stage == 1:
                            color = model_output['color'][j]
                            color = torch.index_select(color.cpu(), 0, idxes).numpy()
                            rgb = color

                        if stage == 4:
                            labels = torch.max(model_output['ans'][0], 2)[1].cpu()

                            all_labels = torch.where(model_output['ans'][1])
                            label = torch.zeros(3000).long().cpu()
                            label[all_labels] = 1
                            label = label * (labels[j] + 1)

                            color = color_list[label]
                            color = torch.index_select(color.cpu(), 0, idxes).numpy()
                            rgb = color

                        if stage in [2,3]:
                            label = torch.argmax(model_output['ans'][1], 1)[j]
                            color = color_list[label]
                            color = torch.index_select(color.cpu(), 0, idxes).numpy()
                            rgb = color

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)

                        if not stage == 0:
                            pcd.colors = o3d.utility.Vector3dVector(rgb)

                        o3d.io.write_point_cloud("%s/%d_predicted.ply"%((cur_vis_dir, model_input['id'][j])), pcd)                                    

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(model_input['point_cloud'][j].cpu().numpy())

                        if stage in [1,2,3,4]:
                            pcd.colors = o3d.utility.Vector3dVector(model_input['rgb'][j].cpu().numpy())

                        o3d.io.write_point_cloud("%s/%d_gt_pcd.ply"%((cur_vis_dir, model_input['id'][j])), pcd)

                        idxes2 = torch.where(gt['occ'][j]  > 0)[0].cpu()
                        points = torch.index_select(model_input['coords'][j].cpu(), 0, idxes2).numpy()

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)

                        if stage in [1,2,3,4]:
                            rgb = torch.index_select(model_input['coord_color'][j].cpu(), 0, idxes2).numpy()
                            pcd.colors = o3d.utility.Vector3dVector(rgb)

                        o3d.io.write_point_cloud("%s/%d_gt_voxel.ply"%((cur_vis_dir, model_input['id'][j])), pcd)

def inference(model, val_dataloader=None):

    if val_dataloader is not None:
        print("Running validation set...")
        with torch.no_grad():
            model.eval()

            m = 0; n = 0;
            for val_i, (model_input, gt) in enumerate(val_dataloader):
                model_input = util.dict_to_gpu(model_input)

                gt = util.dict_to_gpu(gt)

                model_output = model.inference(model_input)

 
