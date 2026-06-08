from pathlib import PosixPath
import copy
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
import sys

from src.scenegraph.utils.spacial_utils import get_maskview_num

from src.scenegraph.utils.slam_utils import MapObjectList, DetectionList
from src.scenegraph.utils.slam import (
    filter_objects,
    resize_gobs,
    create_object_pcd,
    process_pcd,
    get_bounding_box,
)
from src.scenegraph.utils.mapping import (
    compute_spatial_similarities,
)
from src.scenegraph.utils.ious import mask_subtract_contained

CFG = { 'dataset_root': PosixPath('./config/Replica'),
        'dataset_config': PosixPath('./config/replica.yaml'),
        'scene_id': 'room0',
        'start': 0,
        'end': -1,
        'stride': 5,
        'image_height': 680,
        'image_width': 1200,
        'gsa_variant': 'none',
        'detection_folder_name': 'gsa_detections_${gsa_variant}',
        'det_vis_folder_name': 'gsa_vis_${gsa_variant}',
        'color_file_name': 'gsa_classes_${gsa_variant}',
        'device': 'cuda',
        'use_iou': True,
        'spatial_sim_type': 'overlap',
        'phys_bias': 0.0,
        'match_method': 'sim_sum',
        'semantic_threshold': 0.5,
        'physical_threshold': 0.5,
        'sim_threshold': 0.8,
        'sim_threshold_spatial': 0.01,
        'use_contain_number': False,
        'contain_area_thresh': 0.95,
        'contain_mismatch_penalty': 0.5,
        'mask_area_threshold': 25,                      # mask can't be too small
        'mask_conf_threshold': 0.75,                    # mask should have enough high confidence
        'max_bbox_area_ratio': 0.5,
        'skip_bg': False,                               # whether take the background into account
        'bg_classes': ["wall", "floor", "ceiling"],
        'min_points_threshold': 16,
        'downsample_voxel_size': 0.025,
        'dbscan_remove_noise': True,
        'dbscan_eps': 0.1,
        'dbscan_min_points': 10,
        'obj_min_points': 50,                            # the min points in pcd of the obj
        'obj_min_detections': 1,                        # the min detection times of the obj
        'merge_overlap_thresh': 0.7,
        'merge_visual_sim_thresh': 0.8,
        'merge_text_sim_thresh': 0.8,
        'denoise_interval': 20,
        'filter_interval': -1,
        'merge_interval': 20,
        'save_pcd': True,
        'save_suffix': 'overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub',
        'vis_render': False,
        'debug_render': False,
        'class_agnostic': True,
        'save_objects_all_frames': True,
        'render_camera_path': 'replica_room0.json',
        'max_num_points': 512,
        }

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    # Iterate through all detections and merge them into objects
    objects_new = []
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
            objects_new.append(len(objects)-1)
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
            objects_new.append(j.item())
            
    return objects, objects_new

def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k not in ['pcd', 'bbox', 'center']:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    obj1['center'] = np.mean([obj1['center'], obj2['center']],axis=0)
    
    return obj1


class Mapping3d():
    def __init__(self, device):
        self.cfg = DictConfig(CFG)
        # all the objects detected, appropriate objects, objects that need to be updated
        self.objects = MapObjectList(device)
        self.objects_post = MapObjectList(device)
        # the nodes detected in each turn
        self.objects_new_index = []

    def gobs_to_detection_list(
        self,
        cfg, 
        image, 
        depth_array,
        cam_K, 
        idx, 
        gobs, 
        trans_pose = None,
        color_path = None,
        depth_min = 0.2,
        depth_max = 3.5
    ):
        '''
        Return a DetectionList object from the gobs
        All object are still in the camera frame. 
        '''
        fg_detection_list = DetectionList()
        
        gobs = resize_gobs(gobs, image)
        # there is no need to tell the bg
        gobs = self.filter_gobs(gobs, image)
        
        if len(gobs['xyxy']) == 0:
            return fg_detection_list
        
        # Compute the containing relationship among all detections and subtract fg from bg objects
        xyxy = gobs['xyxy']
        masks = gobs['mask']
        gobs['mask'] = mask_subtract_contained(xyxy, masks)
        
        n_masks = len(gobs['xyxy'])
        for mask_idx in range(n_masks):
            mask = gobs['mask'][mask_idx]
            width = cam_K[0][0][2] * 2
            # choose the right camera_matrix
            index = get_maskview_num(mask, width)
            if index < 0:
                continue
            # make the pcd and color it
            camera_object_pcd = create_object_pcd(
                depth_array,
                mask,
                cam_K[index],
                image,
                depth_min,
                depth_max,
                obj_color = None
            )
            
            # It at least contains 5 points
            if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
                continue
            # coordinate transform
            if trans_pose[index] is not None:
                global_object_pcd = camera_object_pcd.transform(trans_pose[index])
            else:
                global_object_pcd = camera_object_pcd
            
            # get largest cluster, filter out noise 
            global_object_pcd = process_pcd(global_object_pcd, cfg)
            
            pcd_bbox = get_bounding_box(cfg, global_object_pcd)
            pcd_bbox.color = [0,1,0]
            
            if pcd_bbox.volume() < 1e-6:
                continue

            points = np.asarray(global_object_pcd.points)
            center = np.mean(points, axis=0)[:-1]
            
            # Treat the detection in the same way as a 3D object
            # Store information that is enough to recover the detection
            fg_detection_list.append({
                'image_idx' : [idx],                             # idx of the image
                'mask_idx' : [mask_idx],                         # idx of the mask/detection
                'color_path' : [color_path],                     # path to the RGB image
                'num_detections' : 1,                            # number of detections in this object
                'mask': [mask],
                'xyxy': [gobs['xyxy'][mask_idx]],
                'conf': [gobs['confidence'][mask_idx]],
                'caption': [gobs['caption'][mask_idx]],
                'caption_conf': [gobs['caption_conf'][mask_idx]],
                'n_points': [len(global_object_pcd.points)],
                'pixel_area': [mask.sum()],
                "inst_color": np.random.rand(3),                 # A random color used for this segment instance
                
                # These are for the entire 3D object
                'pcd': global_object_pcd,
                'center': center, 
                'bbox': pcd_bbox,
            })
        
        return fg_detection_list

    def mapping3d(self, gobs, image_rgb, depth_array, cam_K, pose, depth_min, depth_max):
        depth_array = depth_array[..., 0]
        # stands for grounded SAM observations
        idx = len(self.segment2d_results) - 1

        fg_detection_list = self.gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth_array,
            cam_K = cam_K,
            idx = idx,
            gobs = gobs,
            trans_pose = pose,
            depth_min = depth_min, 
            depth_max = depth_max
        )
            
        if len(fg_detection_list) == 0:
            return
            
        if len(self.objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
                self.objects_new_index.append(i)
            # Skip the similarity computation 
            objects_post_t, objects_new_t = filter_objects(self.cfg, self.objects)
            self.objects_post = copy.deepcopy(objects_post_t)
            self.objects_new_index = [x for x in self.objects_new_index if x in objects_new_t]
            return
        # N_fg * N_objects
        spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
        spatial_sim[spatial_sim < self.cfg.sim_threshold_spatial] = float('-inf')

        # merge the doorway and entrance
        for j, _node in enumerate(fg_detection_list):
            _caption_node = ''.join(_node['caption']).replace(' ','')
            if 'entrance' in _caption_node or 'doorway' in _caption_node:
                center_node = _node['center']
                for k, _object in enumerate(self.objects):
                    _caption_object = ''.join(_object['caption']).replace(' ','')
                    if 'entrance' in _caption_object or 'doorway' in _caption_object:
                        center_object = _object['center']
                        diff = np.sum(np.abs(center_node-center_object))
                        if diff < 1:
                            spatial_sim[j, k] = 1-diff
        
        self.objects, self.objects_new_index = merge_detections_to_objects(self.cfg, 
                                                    fg_detection_list, 
                                                    self.objects, 
                                                    spatial_sim)
        objects_post_t, objects_new_t = filter_objects(self.cfg, self.objects)
        self.objects_post = copy.deepcopy(objects_post_t)
        self.objects_new_index = [x for x in self.objects_new_index if x in objects_new_t]

    def filter_gobs(
        self,
        gobs: dict,
        image: np.ndarray,
    ):
        # If no detection at all
        if len(gobs['xyxy']) == 0:
            return gobs
        
        # Filter out the objects based on various criteria
        idx_to_keep = []
        for mask_idx in range(len(gobs['xyxy'])):
            class_name = gobs['caption'][mask_idx]
            
            # Skip masks that are too small
            if gobs['mask'][mask_idx].sum() < max(self.cfg.mask_area_threshold, 10):
                continue
            
            # Skip the BG classes
            if self.cfg.skip_bg and class_name in self.cfg.bg_classes:
                continue
            
            # Skip the non-background boxes that are too large
            if class_name not in self.cfg.bg_classes:
                x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
                bbox_area = (x2 - x1) * (y2 - y1)
                image_area = image.shape[0] * image.shape[1]
                if bbox_area > self.cfg.max_bbox_area_ratio * image_area:
                    continue
                
            # Skip masks with low confidence
            if gobs['confidence'] is not None:
                if gobs['confidence'][mask_idx] < self.cfg.mask_conf_threshold:
                    continue
            
            idx_to_keep.append(mask_idx)
        
        for k in gobs.keys():
            if isinstance(gobs[k], str) or k == "classes": # Captions
                continue
            elif isinstance(gobs[k], list):
                gobs[k] = [gobs[k][i] for i in idx_to_keep]
            elif isinstance(gobs[k], np.ndarray):
                gobs[k] = gobs[k][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
        
        return gobs
