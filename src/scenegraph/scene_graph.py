import numpy as np
import sys
from typing import Tuple
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from src.scenegraph.utils.slam_utils import MapObjectList
from src.scenegraph.utils.sg_utils import Mapping3d
from src.scenegraph.map import Map

from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from maskrcnn_benchmark.config import cfg as glip_cfg

import io
from ModelServer import ModelClient


class SceneGraph(Mapping3d):
    '''the class of constructing the scenegraph
    '''
    def __init__(self, config, dataset='rxr'):
        self.uni_config = config.UNI
        self.sg_config = config.SCENEGRAPH
        self.doorway_max_depth_diff = config.SCENEGRAPH.DOORWAY_MAX_DEPTH_DIFF
        self.GSAM2_server_port = config.GSAM2_SERVER_PORT
        self.var_threshold = config.SCENEGRAPH.VAR_THRESHOLD
        self.map_cfg = config.MAP
        self.dataset = dataset.lower()  # 'rxr' or 'r2r'

        super().__init__(self.uni_config.DEVICE)

        self.map_draft = Map(config.MAP, config.UNI)

        self.GSAM2 = ModelClient('http://localhost:{}'.format(self.GSAM2_server_port))

        ### ------ init glip model ------ ###
        config_file = "./third_party/GLIP/configs/pretrain/glip_Swin_L.yaml" 
        weight_file = "./third_party/GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            confidence_threshold=0.61,
            show_mask_heatmaps=False
        )

        self.segment2d_results = []
        self.glip_room = None
        self.image_list = []

    def reset(self, dag=None, objects_sum=None):
        # only recognize the object in current episode
        self.dag = None
        if dag:
            self.dag = dag
        elif objects_sum:
            self.text_prompt = ' . '.join(objects_sum) + ' . '
        else:
            self.text_prompt = self.sg_config.TEXT_PROMPT
        
        self.objects = MapObjectList(device=self.uni_config.DEVICE)
        self.objects_post = MapObjectList(device=self.uni_config.DEVICE)
        self.objects_new_index = []
        self.segment2d_results = []
        self.image_list = []
        self.map_draft.reset()

    def get_scenegraph(
            self, 
            stage, 
            image_rgb: np.ndarray, 
            depth_array: np.ndarray, 
            cam_K: np.ndarray, 
            pose: np.ndarray,
            agent_coordinate: np.ndarray, 
            thin_type, 
        ) -> list:
        '''get the object of the text list, return a list of the objects
        '''
        # get the segmentation and caption of a image, store in self.segment2d_results
        result = self.segment2d(image_rgb, depth_array, [stage, stage+1])
        masks = np.array([])
        if result:        
            # get the pcd of the segmentation, store in self.objects_post
            self.mapping3d(
                self.segment2d_results[-1], 
                image_rgb, 
                depth_array, 
                cam_K, 
                pose,
                self.map_draft.cfg.DEPTH_MIN,
                self.map_draft.cfg.DEPTH_MAX,
            )
            # get the pcd of the segmentation, store in self.objects_post
            indices = np.where(np.isin(self.segment2d_results[-1]['caption'], 
                                       self.map_draft.mask_captions))
            masks = self.segment2d_results[-1]['mask'][indices].astype(bool)
        self.map_draft.generate_bev_map(
            depth_array, 
            masks, 
            pose, 
            cam_K[0], 
            agent_coordinate, 
            thin_type,
            self.glip_room,
        )
        self.map_draft.update_node(self.objects, self.objects_new_index)
        self.glip_room = None
        
        return result
    
    def update_target(self, dag, stage):
        '''update the anchor points for verifying according to the stage
        '''
        target_raw = list(dag.neighbors(stage))
        target_t = [item for item in target_raw \
                  if dag.nodes[item]['type'] == 'object']
        object_name = []
        room_name = []
        for item in target_t:
            _name = item.split('_')[0]
            if '*' in _name:
                room_name.append(_name[:-1])
            else:
                object_name.append(_name)
        target_name = object_name + room_name
        return target_name, room_name

    def segment2d(self, image_rgb, depth_array, stages:list):
        if len(depth_array.shape) > 2:
            _depth_array = depth_array.squeeze(-1)
        _depth_mask = (_depth_array>self.map_cfg.DEPTH_MIN * 1.2) & \
                      (_depth_array<self.map_cfg.DEPTH_MAX * 0.9)
        image_rgb_byte_io = io.BytesIO()
        if self.dataset == 'rxr':
            Image.fromarray(image_rgb).save(image_rgb_byte_io, format='JPEG', quality=90)
        else:
            Image.fromarray(image_rgb).save(image_rgb_byte_io, format='PNG')

        if not self.dag:
            text_prompt = self.text_prompt
        else:
            text_list = []
            room_list = []
            for stage in stages:
                _object, _room = self.update_target(self.dag, stage)
                text_list = text_list + _object
                room_list = room_list + _room
            text_prompt = ' . '.join(text_list) + ' . '
            room_prompt = '. '.join(room_list) + '.'
        # ordinary object
        masks, xyxy, captions, masks_conf, captions_conf = \
            self.GSAM2(image_rgb_byte_io, text_prompt)
        masks = np.array(masks)
        xyxy = np.array(xyxy)
        captions = np.array(captions)
        if len(masks.shape) < 3:
            masks = np.expand_dims(masks, axis=0)
        masks_conf = np.array(masks_conf).reshape(-1)
        captions_conf = np.array(captions_conf).reshape(-1)
        for i, _caption in enumerate(captions):
            if 'doorway' in _caption:
                _, masks[i] = self.judge_doorway(masks[i], _depth_array, _depth_mask)
        if 'doorway' in text_prompt or 'entrance' in text_prompt:
            _masks, _xyxy, _captions, _masks_conf, _captions_conf = \
                self.GSAM2(image_rgb_byte_io, 'entrance . ')
            _masks_conf = np.array(_masks_conf).reshape(-1)
            _captions_conf = np.array(_captions_conf).reshape(-1)
            for j, mask in enumerate(_masks):
                result, _mask = self.judge_doorway(mask, _depth_array, _depth_mask)
                if result:
                    try:
                        masks = np.concatenate(
                            (masks, np.expand_dims(_mask, axis=0)), axis=0)
                        xyxy = np.concatenate(
                            (xyxy, np.expand_dims(_xyxy[j], axis=0)), axis=0)
                        captions = np.concatenate(
                            (captions, np.expand_dims(_captions[j], axis=0)), axis=0)
                        masks_conf = np.concatenate(
                            (masks_conf, np.expand_dims(_masks_conf[j], axis=0)), axis=0)
                        captions_conf = np.concatenate(
                            (captions_conf, np.expand_dims(_captions_conf[j], axis=0)), axis=0)
                    except:
                        pass
        if len(room_prompt) > 1:
            result = self.glip_demo.inference(image_rgb[:,:,[2,1,0]], room_prompt)
            if len(result) > 0:
                labels_int = result.get_field("labels").tolist()
                labels = []
                for i in labels_int:
                    if i-1 < len(room_list):
                        labels.append(room_list[i-1])
                bbox = result.bbox.numpy()
                self.glip_room = {'labels':labels, 'bbox':bbox}

        if 0 in masks.shape:
            return False

        self.segment2d_results.append({
            "xyxy": xyxy,
            "confidence": masks_conf,
            "mask": masks.astype(np.uint8),
            "caption_conf": captions_conf,
            "caption": captions,
        })
        self.image_list.append(image_rgb)

        return True

    def judge_doorway(self, mask:np.ndarray, depth_array:np.ndarray, depth_mask:np.ndarray):
        '''whether the mask is of a doorway
        '''
        def dilation(tensor, kernel):
            conv_result = F.conv2d(tensor.unsqueeze(0).unsqueeze(0), 
                                kernel.unsqueeze(0).unsqueeze(0), 
                                padding=kernel.shape[0] // 2)
            
            dilated_tensor = (conv_result > 0).to(torch.float32)
            conv_result = dilated_tensor.squeeze(0).squeeze(0)

            conv_result = conv_result[:tensor.shape[0], :]
            conv_result = conv_result[:, :tensor.shape[1]]
            return conv_result

        def erosion(tensor, kernel):
            conv_result = F.conv2d(tensor.unsqueeze(0).unsqueeze(0), 
                                kernel.unsqueeze(0).unsqueeze(0), 
                                padding=kernel.shape[0] // 2)

            kernel_sum = kernel.sum()
            tolerance = kernel_sum * 0.001
            eroded_tensor = (torch.abs(conv_result - kernel_sum) < tolerance).to(torch.float32)
            conv_result = eroded_tensor.squeeze(0).squeeze(0)

            conv_result = conv_result[:tensor.shape[0], :]
            conv_result = conv_result[:, :tensor.shape[1]]
            return conv_result

        mask_tensor = torch.from_numpy(mask).to(
            device=self.uni_config.DEVICE, dtype=torch.float32)
        depth_tensor = torch.from_numpy(depth_array).to(
            device=self.uni_config.DEVICE, dtype=torch.float32)
        depth_mask_tensor = torch.from_numpy(depth_mask).to(
            device=self.uni_config.DEVICE, dtype=torch.bool)
        
        patch = torch.ones(10, 10, dtype=torch.float32, device=self.uni_config.DEVICE)
        size = int(torch.sqrt(torch.sum(mask_tensor)) * 0.33)
        expansion = torch.ones(size, size, dtype=torch.float32, device=self.uni_config.DEVICE)

        _mask = dilation(mask_tensor, patch)
        _mask = erosion(_mask, patch)
        _mask_dilation = dilation(mask_tensor, expansion)
        _mask = _mask.to(torch.bool)
        _mask_dilation = _mask_dilation.to(torch.bool)

        depth_inner = depth_tensor[_mask & depth_mask_tensor]
        if len(depth_inner) < 4:
            return False, mask
        depth_inner = torch.sort(depth_inner)[0]
        length_in_4 = int(len(depth_inner) // 4)
        depth_inner_t = depth_inner[length_in_4:-length_in_4]
        depth_mean_inner = torch.mean(depth_inner_t)

        mask_outer = _mask_dilation ^ _mask
        depth_outer = depth_tensor[mask_outer & depth_mask_tensor]
        if len(depth_outer) < 4:
            return False, mask
        depth_outer = torch.sort(depth_outer)[0]
        length_out_4 = int(len(depth_outer) // 4)
        depth_outer_t = depth_outer[length_out_4:-length_out_4]
        depth_mean_outer = torch.mean(depth_outer_t)

        if depth_mean_inner - depth_mean_outer > self.doorway_max_depth_diff:
            depth_mask_out = ((depth_tensor < depth_outer_t[-1]) 
                              & (depth_tensor > depth_outer_t[0]))
            result = mask_outer & depth_mask_out
            return True, result.cpu().numpy().astype(int)
        elif depth_mean_outer - depth_mean_inner > self.doorway_max_depth_diff:
            depth_mask_in = ((depth_tensor < depth_inner_t[-1]) 
                              & (depth_tensor > depth_inner_t[0]))
            result = _mask & depth_mask_in
            return True, result.cpu().numpy().astype(int)
        return False, mask
