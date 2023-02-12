import torch

def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict

def occupancy_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict
    
def occupancy_net5(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth.squeeze()

    loss_dict['refer2'] = -1 * (label * torch.log(model_outputs + 1e-5) + (1 - label) * torch.log(1 - model_outputs + 1e-5)).mean()

    return loss_dict

def occupancy_net6(model_outputs, ground_truth):
    loss_dict = dict()
    pred_leg = torch.max(model_outputs['occ_branch'], 1)[0]
    pred_leg = pred_leg.sum(1)
    diff = pred_leg - ground_truth
    loss_dict['cnt'] = torch.sum(diff * diff) / diff.numel()

    return loss_dict

def occupancy_net10(model_outputs, ground_truth):
    loss_dict = dict()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_dict['refer2'] = loss_fn(model_outputs, ground_truth)

    return loss_dict

def occupancy_net11(model_outputs):
    loss_dict = dict()

    gt = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)
    masks = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)

    for j in range(model_outputs['occ_branch'].size()[0]):
        
        occ2 = 50

        pred_leg2 = torch.topk(model_outputs['occ_branch'][j], occ2, 0)[1].transpose(1,0)

        for i in range(3):
            masks[j, i, pred_leg2[i]] = 1
            gt[j, i, pred_leg2[i]] = 1

    diff = (model_outputs['occ_branch'] - gt.permute(0,2,1)) * (masks.permute(0,2,1))

    loss_dict['cnt'] = torch.sum(diff * diff) / diff.numel() * 100

    return loss_dict

def occupancy_net7(model_outputs, ground_truth, occupancy):
    loss_dict = dict()
    pred_leg = torch.max(model_outputs['occ_branch'], 1)[0]

    # pred_leg = pred_leg.sum(1)
    gt = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)
    masks = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)

    for j in range(pred_leg.size()[0]):
        occ = torch.where(occupancy[j] > 0.5)[0].size()[0]
        occ2 = int(occ / (ground_truth[j]))

        pred_leg2 = torch.topk(model_outputs['occ_branch'][j], occ2, 0)[1].transpose(1,0)
        idxes = torch.topk(pred_leg[j], min(ground_truth[j], 6))[1]

        for i in range(6):
            masks[j, i, pred_leg2[i]] = 1
            if i in idxes:
                gt[j, i, pred_leg2[i]] = 1

    diff = (model_outputs['occ_branch'] - gt.permute(0,2,1)) * (masks.permute(0,2,1))

    loss_dict['cnt'] = torch.sum(diff * diff) / diff.numel() * 100

    return loss_dict

def occupancy_net9(model_outputs, ground_truth, occupancy):
    loss_dict = dict()
    pred_leg = torch.max(model_outputs, 1)[0]

    # pred_leg = pred_leg.sum(1)
    gt = torch.zeros_like(model_outputs).permute(0,2,1)
    masks = torch.zeros_like(model_outputs).permute(0,2,1)

    for j in range(pred_leg.size()[0]):
        occ = torch.where(occupancy > 0.5)[0].size()[0]
        # occ = occupancy
        
        # try:
        occ2 = int(occ / (ground_truth[j]))
        # except:
        #     occ2 = 50

        pred_leg2 = torch.topk(model_outputs[j], occ2, 0)[1].transpose(1,0)
        idxes = torch.topk(pred_leg[j], min(ground_truth[j], 10))[1]

        for i in range(10):
            masks[j, i, pred_leg2[i]] = 1
            if i in idxes:
                gt[j, i, pred_leg2[i]] = 1

    diff = (model_outputs - gt.permute(0,2,1)) * (masks.permute(0,2,1))

    loss_dict['cnt'] = torch.sum(diff * diff) / diff.numel() * 100

    return loss_dict

def occupancy_net8(model_outputs, ground_truth, occupancy, labels):
    loss_dict = dict()
    pred_leg = torch.max(model_outputs['occ_branch'], 1)[0]

    gt = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)
    masks = torch.zeros_like(model_outputs['occ_branch']).permute(0,2,1)

    for j in range(pred_leg.size()[0]):
        occ = torch.where(occupancy[j] > 0.5)[0]

        for i in range(6):
            
            idxes = torch.where(labels[j]==i)

            idxes = idxes[0]
            masks[j, i, occ] = 1
            
            gt[j, i, idxes] = 1

    diff = (model_outputs['occ_branch'] - gt.permute(0,2,1)) * (masks.permute(0,2,1))

    loss_dict['cnt'] = torch.sum(diff * diff) / diff.numel() * 100

    return loss_dict

def occupancy_net4(label2, features, gt):

    gt = gt.cpu()
    loss_dict = dict()

    pos = torch.where(label2 < gt)[0]
    neg = torch.where(label2 >= gt)[0]
    pos_features = torch.mean(features[pos])
    neg_features = torch.mean(features[neg])

    loss_dict['ins'] = neg_features - pos_features

    return loss_dict

def occupancy_net2(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    diff = model_outputs - ground_truth
    loss_dict['refer'] = torch.sum(diff * diff) / diff.numel()
    return loss_dict

def occupancy_net3(model_outputs, ground_truth, gt, val=False):
    mask = (gt['occ'] + 1) / 2

    loss_dict = dict()
    diff = model_outputs - ground_truth
    diff = diff*mask
    
    mask_len = torch.where(mask>0)[0].shape[0]

    loss_dict['color'] = torch.sum(diff * diff) / mask_len
    return loss_dict

def distance_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict
