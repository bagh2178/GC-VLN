import torch
import torch.nn.functional as F
import numpy as np
import copy
from skimage import morphology 
from src.solver.math_utils import(
    extract_valid_mask_from_skeleton_bev, 
)

def get_point_cloud_from_z_t(
    depth: np.ndarray, 
    mask: np.ndarray, 
    mask_obj: np.ndarray, 
    camera_matrix: np.ndarray, 
    pose: np.ndarray, 
    labels, 
    room_masks, 
    scene_graph, 
    map_shape, 
    device: str='cuda', 
    scale: int=1, 
    agent_height = 20,
    resolution: int=20
)->torch.tensor:
    """Projects the depth image Y into a 3D point cloud in the camera system.
    Inputs:
        depth: ...xHxW, which consists of N images
        mask: limit the processing scope
        camera_matrix [N]
        device [N]
        pose
        scale: downsampling ratio
        resolution: expansion factor of metric
    Outputs:
        points_coordinate is [M]
    """
    # pretreatment
    depth_tensor = torch.from_numpy(depth.copy()) * resolution
    # to deal with the error that depth has 3th dimension
    if len(depth_tensor.size()) > 2:
        depth_tensor = depth_tensor.squeeze(-1)
    mask_tensor = torch.from_numpy(mask.copy())
    if len(mask_tensor.size()) > 2:
        mask_tensor = mask_tensor.squeeze(-1)
    mask_obj_tensor = torch.from_numpy(mask_obj.copy())
    if len(mask_obj_tensor.size()) > 2:
        mask_obj_tensor = mask_obj_tensor.squeeze(-1)
    # if you change the camera settings, change here as well
    depth_chunks = torch.chunk(depth_tensor, 12, dim=1)
    depth_chunks = torch.stack(depth_chunks, dim=0).to(device)
    mask_chunks = torch.chunk(mask_tensor, 12, dim=1)
    mask_chunks = torch.stack(mask_chunks, dim=0).to(device)
    mask_obj_chunks = torch.chunk(mask_obj_tensor, 12, dim=1)
    mask_obj_chunks = torch.stack(mask_obj_chunks, dim=0).to(device)

    fx = torch.tensor(camera_matrix[0, 0]).to(device)
    fy = torch.tensor(camera_matrix[1, 1]).to(device)
    cy = torch.tensor(camera_matrix[1, 2]).to(device)
    cx = torch.tensor(depth_chunks[0].size()[1] / 2).to(device)

    grid_y, grid_x = torch.meshgrid(torch.arange(depth_chunks[0].shape[-2]),
                                    torch.arange(depth_chunks[0].shape[-1]),
                                )
    grid_x = grid_x.unsqueeze(0).expand(depth_chunks.size()).to(device)
    grid_y = grid_y.unsqueeze(0).expand(depth_chunks.size()).to(device)
    # 12xHxW
    x_t = (grid_x[:,::scale,::scale]-cx) * depth_chunks[:,::scale,::scale] / fx
    y_t = (grid_y[:,::scale,::scale]-cy) * depth_chunks[:,::scale,::scale] / fy
    # 12xHxWx3
    points_coordinate_raw = torch.stack((x_t, y_t, depth_chunks), dim=3)
    # transform pose
    pose_t = pose
    pose_t[:,0:2,3] = pose_t[:,0:2,3] * resolution
    points_coordinate_trans = transformed_coordinate_system(
        points_coordinate_raw,
        pose_t,
        device
    )
    # [N]
    points_coordinate = points_coordinate_trans[mask_chunks]
    points_mask = points_coordinate_trans[mask_obj_chunks]
    points_coordinate[:,-1] += agent_height
    points_mask[:,-1] += agent_height

    # update the room node
    for label, room_mask in zip(labels, room_masks):
        mask_room_chunks = torch.chunk(room_mask, 12, dim=1)
        mask_room_chunks = torch.stack(mask_room_chunks, dim=0).to(device)
        points_room = points_coordinate_trans[mask_room_chunks]
        center = torch.mean(
            points_room[:,:-1]+map_shape//2, dim=0).cpu().numpy().astype(np.int32)
        time = 0
        for node, data in scene_graph.nodes(data=True):
            if data['caption'] == label:
                if np.sum(np.abs(center-data['center'])) < 40:
                    time = -1
                    scene_graph.nodes[node]['center'] = ((center+data['center'])/2).astype(np.int32)
                    scene_graph.nodes[node]['num_detections'] += 1
                    break
                else:
                    time += 1

        if time >= 0:    
            scene_graph.add_node(label+'_'+str(time), center=center, caption=label, num_detections=1)

    return points_coordinate, points_mask

def transformed_coordinate_system(
    points: torch.tensor, 
    pose: np.ndarray, 
    device: str
):
    """Convert camera coordinate system to world coordinate system
    Inputs:
        points: [N*H*W*3]
        pose: [N*4*4]
    Outputs:
        points_coordinate is [N*H*W*3]
    """
    pose_tensor = torch.from_numpy(pose.copy()).to(device)

    # pretreat for matmal
    s1, s2, s3, s4 = points.size()
    points_reshape = points.reshape(s1, -1, s4)
    points_t_homo = torch.cat(
        [points_reshape, torch.ones(s1, s2*s3, 1).to(device)], 
        dim=-1
    )
    points_t_homo = points_t_homo.transpose(1,2)
    points_t = (pose_tensor @ points_t_homo)[:,:-1,:]
    points_t = points_t.transpose(1,2)

    points_t = points_t.reshape(s1, s2, s3, s4)
    return points_t

def splat_feat_nd(grid_size, feat_coor, threshold, device, agent_coordinate):
    '''
    Args:
        grid_size: tensor.size() [y,x].
        feat_coor: [N, 3],
        threshold: the threshold of the weight of each cell of the grid
        device: device
        agent_coordinate: the coordinate of the agent
    Returns:
        grid_detected: 0->detected [y,x],
        grid_obstacle: 1->obstacle [y,x],
        grid_wall: 1->wall [y,x],
    '''
    grid_raw = torch.zeros(grid_size).to(device)
    start_point = torch.tensor(grid_size[0]/2).to(torch.int).to(device)
    # clip
    condition = ((feat_coor[:,0] < start_point-1) & (feat_coor[:,0] > -start_point) &\
                (feat_coor[:,1] < start_point-1) & (feat_coor[:,1] > -start_point)).to(device)
    feat_coor_t = feat_coor[condition]
    pos_coors = feat_coor_t[:,:-1]
    pos_coors = pos_coors.unsqueeze(0).expand(4,-1,2) + start_point
    pos_floors = torch.floor(pos_coors).to(torch.int)
    # obtain the coordinates of four adjacent points
    pos_neigs = copy.deepcopy(pos_floors)
    pos_neigs[1,:,:] += 1
    pos_neigs[2,:,0] += 1
    pos_neigs[3,:,1] += 1
    wts_t = (1 - torch.abs(pos_neigs - pos_coors)).sum(dim=2)
    safe_t = torch.zeros_like(feat_coor_t[:, 2])
    safe_t[(feat_coor_t[:, 2] > 4)&(feat_coor_t[:, 2] < 36)] = 1
    safe_wall = torch.zeros_like(feat_coor_t[:, 2])
    safe_wall[(feat_coor_t[:, 2] > 24)&(feat_coor_t[:, 2] < 36)] = 1
    safe_det = torch.zeros_like(feat_coor_t[:, 2])
    safe_det[(feat_coor_t[:, 2] > -36)&(feat_coor_t[:, 2] < 36)] = 1
    # [4N]
    value_t = (wts_t * safe_t).reshape(-1).to(torch.float)
    value_wall = (wts_t * safe_wall).reshape(-1).to(torch.float)
    value_det = (wts_t * safe_det).reshape(-1).to(torch.float)

    pos_neigs_t = torch.clamp(pos_neigs.reshape(-1, 2), 
                              torch.tensor(0).to(device), 
                              torch.tensor(grid_size[1]-1).to(device))
    # sum, rows correspond to y, columns correspond to x
    pos_neigs_flat = (
        pos_neigs_t[:,1] * grid_size[0] + pos_neigs_t[:,0]
    )
    grid_raw_flat = torch.zeros_like(grid_raw).reshape(-1)
    grid_raw_flat.index_add_(0, pos_neigs_flat, value_t)
    grid_raw_t = grid_raw_flat.reshape(grid_size)

    grid_raw_flat_wall = torch.zeros_like(grid_raw).reshape(-1)
    grid_raw_flat_wall.index_add_(0, pos_neigs_flat, value_wall)
    grid_raw_wall = grid_raw_flat_wall.reshape(grid_size)

    grid_raw_flat_det = torch.zeros_like(grid_raw).reshape(-1)
    grid_raw_flat_det.index_add_(0, pos_neigs_flat, value_det)
    grid_raw_det = grid_raw_flat_det.reshape(grid_size)
    # in gpu
    grid_detected = torch.ones_like(grid_raw).to(device, torch.int)
    grid_obstacle = torch.zeros_like(grid_raw).to(device, torch.int)
    grid_wall = torch.zeros_like(grid_raw).to(device, torch.int)
    kernel_1 = torch.ones(3, 3, dtype=torch.float32, device=device)
    kernel_2 = torch.ones(5, 5, dtype=torch.float32, device=device)
    grid_detected[grid_raw_det >= 1] = 0
    agent_coordinate_t = torch.tensor(agent_coordinate).to(device, dtype=torch.int)
    grid_detected[
        torch.clip(agent_coordinate_t[1]+start_point-25, 0, grid_size[0]-1):
        torch.clip(agent_coordinate_t[1]+start_point+25, 0, grid_size[0]-1), 
        torch.clip(agent_coordinate_t[0]+start_point-25, 0, grid_size[0]-1):
        torch.clip(agent_coordinate_t[0]+start_point+25, 0, grid_size[0]-1)
    ] = 0
    grid_detected = erosion(grid_detected, kernel_2)
    grid_detected = dilation(grid_detected, kernel_1)
    grid_obstacle[grid_raw_t >= threshold] = 1
    grid_obstacle = dilation(grid_obstacle, kernel_2)
    grid_obstacle = erosion(grid_obstacle, kernel_1)
    grid_wall[grid_raw_wall >= threshold] = 1
    grid_wall = dilation(grid_wall, kernel_1)
    return grid_detected, grid_obstacle, grid_wall

def get_white_list(grid_size, feat_coor, device, agent_coordinate):
    '''
    Args:
        grid_size: tensor.size() [y,x].
        feat_coor: [N, 3],
        device: device
        agent_coordinate: the coordinate of the agent
    Returns:
        grid_detected: 0->detected [y,x],
    '''
    grid_raw = torch.zeros(grid_size).to(device)
    start_point = torch.tensor(grid_size[0]/2).to(torch.int).to(device)
    # clip
    condition = ((feat_coor[:,0] < start_point-1) & (feat_coor[:,0] > -start_point) &\
                (feat_coor[:,1] < start_point-1) & (feat_coor[:,1] > -start_point)).to(device)
    feat_coor_t = feat_coor[condition]
    pos_coors = feat_coor_t[:,:-1]
    pos_coors = pos_coors.unsqueeze(0).expand(4,-1,2) + start_point
    pos_floors = torch.floor(pos_coors).to(torch.int)
    # obtain the coordinates of four adjacent points
    pos_neigs = copy.deepcopy(pos_floors)
    pos_neigs[1,:,:] += 1
    pos_neigs[2,:,0] += 1
    pos_neigs[3,:,1] += 1
    wts_t = (1 - torch.abs(pos_neigs - pos_coors)).sum(dim=2)
    safe_det = torch.zeros_like(feat_coor_t[:, 2])
    safe_det[(feat_coor_t[:, 2] > -36)&(feat_coor_t[:, 2] < 36)] = 1
    value_det = (wts_t * safe_det).reshape(-1).to(torch.float)

    pos_neigs_t = torch.clamp(pos_neigs.reshape(-1, 2), 
                              torch.tensor(0).to(device), 
                              torch.tensor(grid_size[1]-1).to(device))
    # sum, rows correspond to y, columns correspond to x
    pos_neigs_flat = (
        pos_neigs_t[:,1] * grid_size[0] + pos_neigs_t[:,0]
    )
    grid_raw_flat_det = torch.zeros_like(grid_raw).reshape(-1)
    grid_raw_flat_det.index_add_(0, pos_neigs_flat, value_det)
    grid_raw_det = grid_raw_flat_det.reshape(grid_size)
    # in gpu
    grid_detected = torch.ones_like(grid_raw).to(device, torch.int)
    kernel_5 = torch.ones(5, 5, dtype=torch.float32, device=device)
    grid_detected[grid_raw_det >= 1] = 0
    grid_detected = dilation(grid_detected, kernel_5)
    return grid_detected

def limit_data_size(datas, mins, maxs, mode=1):
    '''limit the size of data to a certain range [min, max]
    Args: 
        data, min, max [N]
        mode: 0 or 1,Indicate whether to report an error directly 
              or to correct it to a boundary value
    '''
    data_limit = copy.deepcopy(datas)
    results = torch.ones_like(datas)
    for i, [data, min, max] in enumerate(zip(datas, mins, maxs)):
        if data <= max and data >= min:
            continue
        elif data < min:
            results[i] = 0
            if mode:
                data_limit[i] = min
            else:
                raise ValueError("data out of range, too small")
        else:
            results[i] = 0
            if mode:
                data_limit[i] = max
            else:
                raise ValueError("data out of range, too large")
            
    return results, data_limit

def thining(
        thin:torch.Tensor, 
        bev:torch.Tensor, 
        mask_pre:torch.Tensor, 
        mask_connect:torch.Tensor,
        start_point, 
    ):
    '''thining algorithm
    Args:
        thin: the bev map that has been thined,
        bev: the bev map need to be thined,
        mask: the connect area last time, 
        mask_connect: the connect area current time
        start_point: the start point of the path
    '''
    kernel = torch.ones(9, 9, dtype=torch.float32, device=bev.device)
    mask_new = (mask_connect == True) & (mask_pre == False)
    _mask = dilation(mask_new, kernel).to(torch.bool)
    _bev = torch.where(_mask, bev, thin)
    zeros = torch.zeros_like(_bev, dtype=_bev.dtype, device=_bev.device)
    mask = extract_valid_mask_from_skeleton_bev(start_point, _bev)
    _bev = torch.where(mask, _bev, zeros)

    bev_np = _bev.cpu().numpy()
    skeleton = morphology.skeletonize(bev_np)
    
    return torch.from_numpy(skeleton).to(device=thin.device, dtype=thin.dtype)

def thining_v2( 
        bev:torch.Tensor, 
        mask_connect:torch.Tensor,
    ):
    '''thining algorithm
    Args:
        bev: the bev map need to be thined,
        mask_connect: the connect area current time
    '''
    zeros = torch.zeros_like(bev, dtype=bev.dtype, device=bev.device)
    _bev = torch.where(mask_connect, bev, zeros)

    bev_np = _bev.cpu().numpy()
    skeleton = morphology.skeletonize(bev_np)
    
    return torch.from_numpy(skeleton).to(device=bev.device, dtype=bev.dtype)

def dilation(tensor, kernel):
    assert torch.all((tensor == 0) | (tensor == 1)), "Input tensor must be binary (0s and 1s)."
    tensor = tensor.to(torch.float32)
    kernel = kernel.to(torch.float32)
    conv_result = F.conv2d(tensor.unsqueeze(0).unsqueeze(0), 
                           kernel.unsqueeze(0).unsqueeze(0), 
                           padding=kernel.shape[0] // 2)
    
    dilated_tensor = (conv_result > 0).int()
    conv_result = dilated_tensor.squeeze(0).squeeze(0)

    conv_result = conv_result[:tensor.shape[0], :]
    conv_result = conv_result[:, :tensor.shape[1]]
    return conv_result

def erosion(tensor, kernel):
    assert torch.all((tensor == 0) | (tensor == 1)), "Input tensor must be binary (0s and 1s)."
    tensor = tensor.to(torch.float32)
    kernel = kernel.to(torch.float32)
    conv_result = F.conv2d(tensor.unsqueeze(0).unsqueeze(0), 
                           kernel.unsqueeze(0).unsqueeze(0), 
                           padding=kernel.shape[0] // 2)

    kernel_sum = kernel.sum()
    tolerance = kernel_sum * 0.001
    eroded_tensor = (torch.abs(conv_result - kernel_sum) < tolerance).int()
    conv_result = eroded_tensor.squeeze(0).squeeze(0)

    conv_result = conv_result[:tensor.shape[0], :]
    conv_result = conv_result[:, :tensor.shape[1]]
    return conv_result