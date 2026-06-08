import networkx as nx
from src.scenegraph.utils.map_utils import (
    get_point_cloud_from_z_t,
    splat_feat_nd,
    get_white_list, 
    thining, 
    thining_v2, 
)
from src.solver.math_utils import extract_valid_mask_from_skeleton_bev
import numpy as np
import itertools
import torch
import skimage
import networkx as nx

class Map():
    '''Map is used to build the BEV Map and the scene graph
    '''
    def __init__(self, cfg, uni_cfg):
        self.cfg = cfg
        self.device = uni_cfg.DEVICE
        self.start_point = None

        self.bev_map_detected = torch.ones([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_map_obstacle = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_map_wall = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_white_list = torch.ones([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)

        self.bev_map = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_map_fmm = torch.ones([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_wall = torch.ones([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.bev_thin = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.mask_connect = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        
        self.zeros = torch.zeros([cfg.SIZE, cfg.SIZE]).to(self.device, torch.int)
        self.mask_captions = cfg.MASK_CAPTIONS
        self.selem = skimage.morphology.disk(2)
        self.scene_graph = nx.Graph()
        self.ready = None

    def reset(self):
        self.bev_map = torch.zeros([self.cfg.SIZE, 
                                    self.cfg.SIZE]).to(self.device, torch.int)
        self.bev_map_fmm = torch.ones([self.cfg.SIZE, 
                                       self.cfg.SIZE]).to(self.device, torch.int)
        self.bev_wall = torch.ones([self.cfg.SIZE, 
                                        self.cfg.SIZE]).to(self.device, torch.int)
        self.bev_thin = torch.zeros([self.cfg.SIZE, 
                                     self.cfg.SIZE]).to(self.device, torch.int)
        
        self.bev_map_detected = torch.ones(
            [self.cfg.SIZE, self.cfg.SIZE]
        ).to(self.device, torch.int)
        self.bev_white_list = torch.ones(
            [self.cfg.SIZE, self.cfg.SIZE]
        ).to(self.device, torch.int)
        self.bev_map_obstacle = torch.zeros(
            [self.cfg.SIZE, self.cfg.SIZE]
        ).to(self.device, torch.int)
        self.bev_map_wall = torch.zeros(
            [self.cfg.SIZE, self.cfg.SIZE]
        ).to(self.device, torch.int)
        self.mask_connect = torch.zeros(
            [self.cfg.SIZE, self.cfg.SIZE]
        ).to(self.device, torch.int)
        self.scene_graph = nx.Graph()

    def generate_bev_map(
        self, 
        depth_image: np.ndarray, 
        masks: np.ndarray,
        pose: np.ndarray,
        camera_matrix,
        agent_coordinate: np.ndarray,
        thin_type, 
        rooms, 
    ):
        '''Generate the bev map, or 2D occupancy map
        '''
        mask, mask_obj = self.generate_mask(depth_image, masks)

        # get the room info
        labels = []
        bbox = []
        if rooms is not None:
            labels = rooms['labels']
            bbox = rooms['bbox']
        room_masks = []
        for x1, y1, x2, y2 in bbox:
            room_mask = torch.zeros(mask.shape[0], mask.shape[1]).to(torch.bool)
            room_mask[int(y1):int(y2), int(x1):int(x2)] = True
            room_masks.append(room_mask)
        
        points, points_mask = get_point_cloud_from_z_t(
            depth_image, 
            mask, 
            mask_obj, 
            camera_matrix, 
            pose, 
            labels, 
            room_masks, 
            self.scene_graph, 
            self.cfg.SIZE, 
            self.device, 
            self.cfg.SCALE,
            self.cfg.AGENT_HEIGHT,
            self.cfg.RESOLUTION
        )

        mask_detected, mask_obstacle, mask_wall = splat_feat_nd(
            self.bev_map.size(), 
            points, 
            self.cfg.PASS_THRESHOLD,
            self.device,
            np.array(agent_coordinate) * self.cfg.RESOLUTION
        )

        self.bev_map_obstacle = self.bev_map_obstacle | mask_obstacle
        self.bev_map_wall = self.bev_map_wall | mask_wall 
        self.bev_map_detected = self.bev_map_detected & mask_detected 

        pre_mask_connect = self.mask_connect
        self.bev_map = 1 - (self.bev_map_detected | self.bev_map_obstacle)
        self.bev_wall = 1 - self.bev_map_wall
        self.bev_map_fmm = 1 - self.bev_map_obstacle

        if self.start_point is None:
            start_point = [int(self.cfg.SIZE/2), int(self.cfg.SIZE/2)]
        else:
            start_point = self.start_point
        self.mask_connect = extract_valid_mask_from_skeleton_bev(
            start_point, 
            self.bev_map
        )
        self.bev_thin = torch.where(self.mask_connect, self.bev_thin, self.zeros)
        if thin_type == 1:
            self.bev_thin = thining(
                self.bev_thin, 
                self.bev_map, 
                pre_mask_connect, 
                self.mask_connect,
                start_point
            )
        else:
            self.bev_thin = thining_v2(self.bev_map, self.mask_connect)
        self.ready = self.find_nearest_point(self.bev_thin)
        if self.ready:
            self.start_point = self.ready

    def update_node(self, objects, index):
        '''
        Args:
        objects: all the objects before filter
        index: index of the new objects
        '''
        # update nodes
        for i in index:
            # update nodes
            if not i in self.scene_graph:
                self.scene_graph.add_node(i)
            # update the caption, confidence, pcd 2d center, pcd 2d scope in WCS
            captions = np.unique(objects[i]['caption'])
            caption_conf = np.array([])
            for c in captions:
                mask = np.where(np.array(objects[i]['caption']) == c)[0]
                caption_conf = np.append(
                    caption_conf, np.mean(np.array(objects[i]['caption_conf'])[mask])
                )
            index_t = np.argmax(caption_conf)
            caption_t = captions[index_t]
            caption_conf_t = caption_conf[index_t]
            self.scene_graph.nodes[i]['caption'] = caption_t
            self.scene_graph.nodes[i]['caption_conf'] = caption_conf_t
            self.scene_graph.nodes[i]['num_detections'] = objects[i]['num_detections']
            points = np.asarray(objects[i]['pcd'].points)
            self.scene_graph.nodes[i]['center'] = (
                objects[i]['center'] * self.cfg.RESOLUTION + self.cfg.SIZE / 2).astype(int)
            self.scene_graph.nodes[i]['scope'] = \
                np.array([np.max(points, axis=0)[:-1] * self.cfg.RESOLUTION + self.cfg.SIZE / 2, 
                          np.min(points, axis=0)[:-1] * self.cfg.RESOLUTION + self.cfg.SIZE / 2]).astype(int)
        # update edges
        combination_list = list(itertools.combinations(index, 2))
        self.scene_graph.add_edges_from(combination_list)

    def generate_mask(self, depth_image: np.ndarray, masks: np.ndarray=np.array([])):
        '''
        Generate a mask using True to indicate the region that can 
        be processed

        Args:
            depth_image,
            mask: it uses True to indicate the region of some passable object
        '''
        mask_t = (depth_image > 1.1 * self.cfg.DEPTH_MIN) & \
                 (depth_image < 0.9 * self.cfg.DEPTH_MAX)
        # to deal with the error that depth has 3th dimension
        mask_t = np.squeeze(mask_t, axis=2)
        mask_object = np.ones_like(mask_t)
        if masks.size > 0:
            for mask in masks:
                mask_object = mask_object & (~mask)

        mask_object = skimage.morphology.binary_closing(mask_object, self.selem)
        mask_t = mask_t & mask_object
        return mask_t, ~mask_object
    
    def find_nearest_point(self, skeleton_bev):
        '''find an equivalent starting point on the bev_thin graph
        '''
        ys, xs = torch.where(skeleton_bev)
        raw_point = torch.stack([ys, xs], dim=1)
        if len(raw_point) == 0:
            return None
        skeleton_pts = raw_point.float()
        target = torch.tensor([self.cfg.SIZE//2, self.cfg.SIZE//2], device=skeleton_pts.device)
        deltas = skeleton_pts - target
        sq_distances = torch.sum(deltas ** 2, dim=1)
        min_idx = torch.argmin(sq_distances)
        nearest_pt = raw_point[min_idx].long().tolist()

        return nearest_pt
        