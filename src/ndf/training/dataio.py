import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import json
import os.path as osp
from scipy.spatial.transform import Rotation
import pickle
import os
import shutil

from ndf_robot.utils import path_util, geometry


class JointOccTrainDataset(Dataset):
    def __init__(self, sidelength, depth_aug=False, multiview_aug=False, phase='train', stage=0, vis=False, test=False):

        self.stage = stage
        self.vis = vis
        self.test = test

        path = "./data/shapes"

        self.ques_dict = json.load(open("./data/%s_questions.json"%phase))

        self.files = []
        self.ques = []
        self.filenames = []

        filenames = json.load(open("data/%s.json"%phase))
        files = [os.path.join(path, f) for f in json.load(open("data/%s.json"%phase))]

        for (file, filename) in zip(files, filenames):
            data = np.load(file, allow_pickle=True)

            if ((stage in [0,1]) or vis) and not test:
                self.files.append(file)
                self.filenames.append(filename)
                continue

            questions, answers = self.ques_dict[filename]["programs"], self.ques_dict[filename]["prog_answers"]

            for (idx, ques) in enumerate(questions):
                if ((not stage in [0,1]) and not vis):
                    if not len(ques) == 2: continue
                    if stage == 2:
                        if not (ques[0][0] in [1] and ques[1][0] == -1) :
                            continue
                    if stage == 3:
                        if not ques[0][0] in [1]:
                            continue
                    if stage == 4:
                        if ques[0][0] in [0]:
                            continue

                self.files.append(file)
                self.filenames.append(filename)
                self.ques.append(idx)

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug

        block = 128 
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        filename = self.filenames[index]

        if ((not self.stage in [0,1]) and not self.vis) or self.test:
            q_index = self.ques[index]

        data = np.load(self.files[index], allow_pickle=True)
        posecam =  data['object_pose_cam_frame']  # legacy naming, used to use pose expressed in camera frame. global reference frame doesn't matter though

        idxs = list(range(posecam.shape[0]))
        random.shuffle(idxs)
        select = random.randint(1, 4)

        if self.multiview_aug:
            idxs = idxs[:select]

        poses = []
        quats = []
        for i in idxs:
            pos = posecam[i, :3]
            quat = posecam[i, 3:]

            poses.append(pos)
            quats.append(quat)

        depths = []
        segs = []
        rgbs = []
        cam_poses = []

        for i in idxs:
            seg = data['object_segmentation'][i]
            depth = data['depth_observation'][i]
            rgb = data['rgb_observation'][i]

            rix = np.random.permutation(depth.shape[0])[:1000]
            seg = seg[rix]
            depth = depth[rix]
            rgb = rgb[rix]

            if self.depth_aug:
                depth = depth + np.random.randn(*depth.shape) * 0.1

            segs.append(seg)
            rgbs.append(torch.from_numpy(rgb))
            depths.append(torch.from_numpy(depth))
            cam_poses.append(data['cam_pose_world'][i])

        # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
        y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

        # Compute native intrinsic matrix
        sensor_half_width = 320
        sensor_half_height = 240

        vert_fov = 60 * np.pi / 180

        vert_f = sensor_half_height / np.tan(vert_fov / 2)
        hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

        intrinsics = np.array(
            [[hor_f, 0., sensor_half_width, 0.],
            [0., vert_f, sensor_half_height, 0.],
            [0., 0., 1., 0.]]
        )

        # Rescale to new sidelength
        intrinsics = torch.from_numpy(intrinsics)

        # build depth images from data
        dp_nps = []

        for i in range(len(segs)):
            seg_mask = segs[i]
            dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :], rgbs[i])

            dp_np = torch.cat([dp_np[:,:3], torch.ones_like(dp_np[..., :1]), dp_np[:,3:]], dim=-1)
            dp_nps.append(dp_np)

        coord, voxel_bool, coord_color = pickle.load(open("./data/parse_dicts/%s_.pkl"%filename, "rb"))

        non_zero = np.where(voxel_bool.squeeze() == True)[0]
        
        rix1 = np.random.permutation(non_zero.shape[0])
        
        non_zero = non_zero[rix1[:1500]]

        zero = np.where(voxel_bool.squeeze() == False)[0]
        rix2 = np.random.permutation(zero.shape[0])
        zero = zero[rix2[:(3000 - non_zero.shape[0])]]

        coord = np.concatenate([coord[non_zero], coord[zero]])

        label = np.concatenate([voxel_bool[non_zero], voxel_bool[zero]])

        coord_color = np.concatenate([coord_color[non_zero], coord_color[zero]])

        coord = torch.from_numpy(coord)
        coord_color = torch.from_numpy(coord_color)
        # transform everything into the same frame
        transforms = []
        for quat, pos in zip(quats, poses):
            quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, -1] = pos
            transform = torch.from_numpy(transform)
            transforms.append(transform)

        idxes = torch.where(torch.from_numpy(label)>0)[0]

        transform = transforms[0]

        coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)

        coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)

        coord = coord[..., :3]
        
        points_world = []
        rgb_world = []

        for i, dp_np in enumerate(dp_nps):
            point_transform = torch.matmul(transform, torch.inverse(transforms[i]))

            rgb_world.append(dp_np[..., 4:]/255)

            dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :4], dim=-1)
            points_world.append(dp_np[..., :3])            

        rgb_world = torch.cat(rgb_world, dim=0)
        point_cloud = torch.cat(points_world, dim=0)

        rix = torch.randperm(point_cloud.size(0))
        point_cloud = point_cloud[rix[:1000]]
        rgb_world = rgb_world[rix[:1000]]

        if point_cloud.size(0) != 1000:
            return self.get_item(index=random.randint(0, self.__len__() - 1))

        label = (label - 0.5) * 2.0

        # translate everything to the origin based on the point cloud mean
        center = point_cloud.mean(dim=0)
        coord = coord - center[None, :]
        point_cloud = point_cloud - center[None, :]

        labels = label
        
        if (not self.stage in [0, 1]) or self.test:
            if self.vis and self.stage == 2:
                question = [[2, -1], [-1, -1]]
                answer = -1
            elif self.vis and self.stage == 3:
                question = [[3, -1], [-1, -1]]
                answer = -1
            elif self.vis and self.stage == 4:
                question = [[3, -1], [4, 2]]
                answer = -1
            else:
                question = self.ques_dict[filename]["programs"][q_index]
                answer = int(self.ques_dict[filename]["prog_answers"][q_index])

            question = torch.tensor(question)

        # # at the end we have 3D point cloud observation from depth images, voxel occupancy values and corresponding voxel coordinates                     

        res = {'point_cloud': point_cloud.float(),
                'rgb': rgb_world.float(),
                'coords': coord.float(),
                'intrinsics': intrinsics.float(),
                'cam_poses': np.zeros(1),
                'occ': torch.from_numpy(labels).float(),
                'coord_color': coord_color.float(),
                'id': index
                }  # cam poses not used

        if (not self.stage in [0,1]) or self.test:
            res['question'] = question
            res['answer'] = answer

        return res, {'occ': torch.from_numpy(labels).float()}

    def __getitem__(self, index):
        return self.get_item(index)
