import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import itertools
import random
from src.solver.constraint import Constraint
from src.solver.navigation_tree import NavigationTree
from src.solver.math_utils import(
    edt, 
    WeightedKMeans, 
    extract_valid_mask_from_skeleton_bev, 
) 
from scipy.ndimage import label, sum_labels
from copy import deepcopy


class Region_Solver():
    '''utilize the mask of constraint and bev to select waypoints
    '''
    def __init__(self, cfg, device='cuda', dataset='rxr'):
        self.config = cfg.RS
        self.radius = cfg.RS.RADIUS
        self.detect_area = cfg.RS.DETECTAREA
        self.device = device
        self.safe_distance = cfg.RS.SAFEDISTANCE
        self.max_distance = cfg.RS.MAXDISTANCE
        self.type_weight = cfg.RS.TYPEWEIGHT
        self.dis_interval = cfg.RS.DIS_INTERVAL
        self.resolution = cfg.MAP.RESOLUTION
        self.map_size = cfg.MAP.SIZE
        self.map_min_size = torch.tensor(0, device=self.device)
        self.map_max_size = torch.tensor(self.map_size-1, device=self.device)
        self.num_detections_threshold = cfg.RS.NUM_DETECTIONS_THRESHOLD
        self.min_region_threshold = cfg.RS.MIN_REGION_THRESHOLD
        self.random_explore_threshold = cfg.RS.RANDOM_EXP_TIME
        self.random_radius = cfg.RS.RANDOM_RADIUS
        self.exp_threshold_one_stage = cfg.RS.EXP_THRESHOLD_ONE_STAGE
        self.dataset = dataset.lower()

    def update_info(self, instruction_dag: nx.DiGraph, end_stage):
        '''update instruction for each episode
        '''
        self.thin_or_not = 0
        self.best_point = np.array([500, 500])
        self.best_sequence = 0
        self.new_stage = False
        self.navigation_mode = 'direction'
        self.random_explore_time = 0
        self.exp_time_this_stage = 0
        self.pre_stage = 1
        self.stage  = 1
        self.final_pts = {}
        self.pre_point = np.array([500, 500])
        self.first_point = None
        self.first_angle = None
        self.end_stage = end_stage
        self.instruction_dag = instruction_dag
        self.navigation_tree = NavigationTree(
            instruction_dag, 
            end_stage, 
            self.radius,
            self.device
        )
        self.constraints = {}
        self.masks = {}

        # 0 - normal exploration, 1 - advanced exploration
        self.explore_state = 0
        self.explore_times = 0

    def node_filter(self, scene_graph, agent_location):
        '''filt the unproper node
        Return:
        the dict {node:data} of proper nodes
        '''
        dict = {}
        for node, data in scene_graph.nodes(data=True):
            # distance
            distance = np.sqrt(
                (data['center'][1] - agent_location[0])**2 + \
                (data['center'][0] - agent_location[1])**2
            )
            if distance <= self.detect_area:
                dict[node] = data

        return dict

    def update_target(self, stage=None):
        '''update the anchor points for verifying according to the stage
        Return:
        target_t: names in the instruction graph, list
        target_name: names in the scene graph, np.array
        '''
        if stage is None:
            _stage = self.stage
        else:
            _stage = stage
        target_raw = list(self.instruction_dag.neighbors(_stage))
        target_t = [item for item in target_raw \
                  if self.instruction_dag.nodes[item]['type'] == 'object']
        target_name = np.array([item.split('_')[0] for item in target_t])
        return target_t, target_name
    
    def get_point_through(self, object_center:list, source_location:list, stage):
        '''get the proper point for the constraint of 'through' and 'weave'
        '''
        obj_center_tensor = torch.tensor(
            object_center, device=self.device, dtype=torch.float32).unsqueeze(0)
        node_location = torch.tensor([source_location], device=self.device, dtype=torch.float32)
        for _, data in self.navigation_tree.waypoints_tree.nodes(data=True):
            if data['stage'] == stage:
                _location = data['location'].unsqueeze(0)
                node_location = torch.cat((node_location, _location), dim=0)
        dis_node_object = torch.sum(torch.abs(node_location-obj_center_tensor), dim=1)
        dis_node_source = torch.sum(torch.abs(node_location-node_location[0]), dim=1)
        dis_object_source = torch.sum(torch.abs(obj_center_tensor-node_location[0]), dim=1)

        index_nodes = torch.where(dis_node_source < dis_object_source)[0]
        negative_dis = -dis_node_object[index_nodes]
        _, index_node = torch.topk(negative_dis, k=min(2, len(negative_dis)))

        result = torch.mean(node_location[index_nodes[index_node]], dim=0)

        return result.cpu().tolist()
    
    def judge_node_valid(self, node, relation, stage):
        '''judge whether the node of the same relation has been used 
        '''
        if not node in self.navigation_tree.navigation_tree:
            return False
        for _, v, data in self.navigation_tree.navigation_tree.out_edges(node, data=True):
            if data['relation'] == relation:
                if v in self.navigation_tree.path and \
                    self.navigation_tree.navigation_tree.nodes[v]['stage'] < stage:
                    return True
                else:
                    return False
        return False

    def update_instance(self, agent_location, scene_graph: nx.Graph, stage=None):
        '''update the corresponding relation of object instance and navigation tree
        '''
        mask_t = {}
        if stage is None:
            _stage = self.stage
            # get the matching target
            target_t, target_name = self.update_target(_stage)
        else:
            _stage = stage
            _, target_name_pre = self.update_target(_stage-1)
            target_t, target_name = self.update_target(_stage)
            if all(element in target_name_pre for element in target_name):
                return None
        
        target_name = np.array([item.replace(' ', '') for item in target_name])
        for target in target_t:
            mask_t[target] = {
                'relation': self.instruction_dag[_stage][target]['relation'],
                'nodes': [],
                'constraint': []
            }
        # get the corresponding object node for each matching target
        dict_nodes = self.node_filter(scene_graph, agent_location)
        for node, data in dict_nodes.items():
            if data['num_detections'] >= self.num_detections_threshold:
                indices = np.array([])
                for index, element in enumerate(target_name):
                    _target = target_t[int(index)]
                    if self.judge_node_valid(
                        node, 
                        self.instruction_dag[_stage][_target]['relation'], 
                        _stage
                    ):
                        continue
                    _caption = data['caption'].replace(' ', '')
                    if 'entrance' in _caption:
                        if 'doorway' in element:
                            indices = np.append(indices, index)
                            continue
                    if _caption in element or element in _caption:
                        indices = np.append(indices, index)
                for index in indices:
                    target = target_t[int(index)]
                    mask_t[target]['nodes'].append(node)

        # if get point of next stage but matched object is not enough
        target_matched = 0
        for _, value in mask_t.items():
            if len(value['nodes']) > 0:
                target_matched += 1
        if (stage is not None) and \
            target_matched < len(target_t) * self.navigation_tree.min_match_ratio - 0.1:
            return None
        # set the constraint
        weave_dict = {}
        for obj, value in mask_t.items():
            relation = value['relation']
            if relation != 'weave':
                for node in value['nodes']:
                    center = scene_graph.nodes[node]['center']
                    if relation in ["right", "left", "back", "front"]:
                        args = [[center[1], center[0]]]
                        # object related to the agent
                        if relation == 'right':
                            angle_direction = -np.pi/2+np.pi
                        elif relation == 'left':
                            angle_direction = np.pi/2+np.pi
                        elif relation == 'back':
                            angle_direction = np.pi+np.pi
                        else:
                            angle_direction = np.pi
                    elif relation in ["near", "pass"]:
                        args = [[center[1], center[0]]]
                        angle_direction = 0
                    elif relation in ["through_f"]:
                        for i in range(_stage, 0, -1):
                            if i in self.navigation_tree.stage_begin:
                                point = self.get_point_through([center[1], center[0]], self.navigation_tree.stage_begin[i], i)
                                break
                        angle_direction = 0
                        args = [[center[1], center[0]], point]
                    else:
                        continue
                    value['constraint'].append(
                        Constraint.set_constraint(
                            "relation", relation, angle_direction, args, dataset=self.dataset
                        )
                    )
            else:
                if len(value['nodes']) > 0:
                    weave_dict[obj] = deepcopy(value)
                    mask_t[obj]['nodes'] = []

        if len(weave_dict) > 0:
            node_list = []
            obj_list = []
            angle_direction = 0
            for obj, value in weave_dict.items():
                obj_list.append(obj)
                node_list.append(value['nodes'])
            for combination in itertools.product(*node_list):
                center = np.array([0,0])
                for node in combination:
                    center += scene_graph.nodes[node]['center']
                center = center / len(combination)
                for i in range(_stage, 0, -1):
                    if i in self.navigation_tree.stage_begin:
                        point = self.get_point_through([center[1], center[0]], self.navigation_tree.stage_begin[i], i)
                        args = [[center[1], center[0]], point]
                        break
                node_visiable = []
                # the nodes of weave must be visiable for each other
                for a, b in itertools.combinations(combination, 2):
                    result = scene_graph.has_edge(a, b)
                    _a = scene_graph.nodes[a]['center']
                    _b = scene_graph.nodes[b]['center']
                    result = (result and np.linalg.norm(_a - _b) < 100)
                    node_visiable.append(result)
                # is weave failed, choose the closest one
                if False in node_visiable or len(node_visiable) == 0:
                    dis = []
                    for node in combination:
                        dis.append(
                            np.linalg.norm(
                                scene_graph.nodes[node]['center'] - np.array(agent_location)
                            )
                        )
                    combination = [combination[np.argmin(dis)]]
                    args.append(
                        [scene_graph.nodes[combination[0]]['center'][1],
                            scene_graph.nodes[combination[0]]['center'][0]]
                    )
                else:
                    for node in combination:
                        args.append(
                            [scene_graph.nodes[node]['center'][1],
                                scene_graph.nodes[node]['center'][0]]
                        )

                _constraint = Constraint.set_constraint(
                                "relation", relation, angle_direction, args, dataset=self.dataset
                            )
                for index, node in enumerate(combination):
                    mask_t[obj_list[index]]['nodes'].append(node)
                    mask_t[obj_list[index]]['constraint'].append(_constraint)
        if stage is not None:
            return mask_t
        else:
            self.constraints[_stage] = mask_t

    def local_max(self, distance_map):
        max_pool = F.max_pool2d(
            distance_map.unsqueeze(0).unsqueeze(0), 
            kernel_size=2*self.radius+1,
            padding=self.radius,
            stride=1
        )
        local_max = (distance_map == max_pool.squeeze())
        y, x = torch.where(distance_map >= self.max_distance)
        for y_t, x_t in zip(y, x):
            length = distance_map[y_t, x_t].to(torch.int32) - self.safe_distance
            local_max[y_t-length:y_t+length, x_t-length:x_t+length] = 1

        return local_max

    def region_select(
            self, 
            bev:torch.tensor, 
            angle_agent, 
            agent_location, 
            oc_map,
            bev_thin=None,
        ):
        '''get potential suitable areas according to the mask and bev
        '''
        masks = []
        keys = []
        # get the navigation direction mask
        shape = bev.shape
        dag_edge_stage = self.instruction_dag[self.stage][self.stage+1]
        angle_agent_t = angle_agent + dag_edge_stage['position2next']

        if self.navigation_mode == 'direction':
            direction = dag_edge_stage['direction']
            position2next = dag_edge_stage['position2next']
        else:
            direction = 'unknown'
            position2next = 0
        nav_constraint = Constraint.set_constraint(
            "navigation_direction",
            direction,
            position2next,
            [[agent_location[0], agent_location[1]]],
            dataset=self.dataset
        )
        nav_mask = nav_constraint.draw_mask(shape, angle_agent, oc_map)

        if self.thin_or_not:
            valid_mask = nav_mask & bev.to(torch.bool)
        else:
            # calculate the distance map
            distance_map = edt(bev).to(self.device)
            valid_mask = (
                nav_mask & (distance_map >= self.safe_distance)
            )
            if bev_thin is not None:
                valid_mask = valid_mask & bev_thin.to(torch.bool)
        for key, mask in self.constraints[self.stage].items():
            for i, constraint in enumerate(mask['constraint']):
                _navigation_direction = angle_agent_t
                if direction == 'unknown' and constraint.type != 'near':
                    # guide navigation direction by objects
                    _c = torch.tensor(constraint.center)
                    _a = torch.tensor(agent_location)
                    _angle_agent = torch.atan2(_c[0]-_a[0], _c[1]-_a[1])
                    _navigation_direction = _angle_agent
                masks.append(constraint.draw_mask(shape, _navigation_direction, oc_map))
                keys.append([key, i])
        
        # generate the type of combination type
        type_tensor = torch.zeros_like(nav_mask, dtype=torch.int32, device=self.device)
        for idx, mask in enumerate(masks):
            if idx >= 63:
                break
            type_tensor = torch.where(mask, type_tensor | (1 << idx), type_tensor)

        self.masks[self.stage] = {
                'nav_mask': nav_mask,
                'object_masks': masks,
                'valid_mask': valid_mask,
                'type_tensor': type_tensor,
                'keys': keys,
            }
        
    def adaptive_cluster_num(
            self, 
            candidates:torch.tensor,
            type_num
        ):
        map_t = torch.zeros(self.map_size, self.map_size)
        row_indexes = candidates[:,0].to(torch.long)
        column_indexes = candidates[:,1].to(torch.long)
        map_t[row_indexes, column_indexes] = 1
        num_cluster = 0
        while 1 in map_t:
            rows, columns = torch.where(map_t == 1)
            index = 0
            num_cluster += 1
            x0 = torch.clamp(rows[index]-self.radius, 0, self.map_size)
            x1 = torch.clamp(rows[index]+self.radius, 0, self.map_size)
            y0 = torch.clamp(columns[index]-self.radius, 0, self.map_size)
            y1 = torch.clamp(columns[index]+self.radius, 0, self.map_size)
            map_t[x0:x1, y0:y1] = 0
            
        n_clusters = max(type_num, num_cluster)
        n_clusters = min(n_clusters, candidates.size(0))
        return n_clusters

    def remove_small_regions(
            self, 
            tensor: torch.Tensor, 
            threshold: int, 
        ) -> torch.Tensor:
        """Remove connected regions with an area smaller than the threshold in 2-channel binary images
        """
        assert tensor.ndim == 2, "The input must be a single channel 2D tensor (H, W)"
        device = tensor.device
        np_arr = tensor.cpu().numpy().astype(np.int8)
        
        struct = np.ones((3,3), dtype=np.int8)
        # label the connected regions
        labeled, num_features = label(np_arr, structure=struct)
        if num_features == 0:
            return tensor
        # calculate the area of each region
        sizes = sum_labels(np_arr, labeled, index=range(1, num_features+1))      
        # create mask
        threshold_t = np.min((threshold, np.sum(np_arr) / 10))
        valid_labels = np.where(sizes >= threshold_t)[0] + 1
        mask = np.isin(labeled, valid_labels)
        
        return torch.from_numpy(mask).to(device=device, dtype=torch.uint8)

    def point_set(self, bev):
        '''extract points from areas
        '''
        valid_mask = deepcopy(self.masks[self.stage]['valid_mask'])
        valid_mask_t = deepcopy(valid_mask)
        type_tensor = self.masks[self.stage]['type_tensor']
        object_masks = self.masks[self.stage]['object_masks']
        # Get all candidate points coordinates
        if not self.thin_or_not:
            valid_mask = self.remove_small_regions(valid_mask, self.min_region_threshold)
        if self.thin_or_not != 2:
            if self.stage in self.final_pts:
                for point in self.final_pts[self.stage]:
                    x0 = torch.max(point[0]-self.radius, self.map_min_size).to(torch.long)
                    x1 = torch.min(point[0]+self.radius, self.map_max_size).to(torch.long)
                    y0 = torch.max(point[1]-self.radius, self.map_min_size).to(torch.long)
                    y1 = torch.min(point[1]+self.radius, self.map_max_size).to(torch.long)
                    valid_mask[x0:x1, y0:y1] = False
        else:
            for point in self.navigation_tree.point_choose[self.stage]:
                x0 = torch.max(point[0]-self.radius, self.map_min_size).to(torch.long)
                x1 = torch.min(point[0]+self.radius, self.map_max_size).to(torch.long)
                y0 = torch.max(point[1]-self.radius, self.map_min_size).to(torch.long)
                y1 = torch.min(point[1]+self.radius, self.map_max_size).to(torch.long)
                valid_mask[x0:x1, y0:y1] = False
        valid_mask[valid_mask_t & (type_tensor>0)] = True
        if self.thin_or_not != 2:
            for point, data in self.navigation_tree.waypoints_tree.nodes(data=True):
                if data['stage'] == self.stage and len(data['match_result']) > 0:
                    x, y = data['location']
                    x0 = torch.max(x-self.radius, self.map_min_size).to(torch.long)
                    x1 = torch.min(x+self.radius, self.map_max_size).to(torch.long)
                    y0 = torch.max(y-self.radius, self.map_min_size).to(torch.long)
                    y1 = torch.min(y+self.radius, self.map_max_size).to(torch.long)
                    valid_mask[x0:x1, y0:y1] = False
        else:
            for point in self.navigation_tree.object_detect[self.stage]:
                x, y = point
                x0 = torch.max(x-self.radius, self.map_min_size).to(torch.long)
                x1 = torch.min(x+self.radius, self.map_max_size).to(torch.long)
                y0 = torch.max(y-self.radius, self.map_min_size).to(torch.long)
                y1 = torch.min(y+self.radius, self.map_max_size).to(torch.long)
                valid_mask[x0:x1, y0:y1] = False
        valid_mask = valid_mask & self.navigation_tree.existed_point_mask

        y, x = torch.where(valid_mask)
        candidates = torch.stack([y, x], dim=1).float()
        if candidates.size(0) == 0:
            return torch.empty((0,2), device=self.device)

        # Get type features (binary decomposition)
        type_features = type_tensor[y, x]
        if len(object_masks) == 0:
            type_num = 1
            type_bits = type_features.unsqueeze(-1).float()
        else:
            bit_indices = torch.arange(len(object_masks), device=self.device)
            type_bits = ((type_features.unsqueeze(-1) >> bit_indices) & 1).float()
            type_num = 1 + len(torch.unique(type_bits, dim=0))

        # Build composite feature space
        h, w = bev.shape
        coord_features = candidates / torch.tensor([[h, w]], device=self.device)
        features = torch.cat([
            coord_features,                  # Normalized coordinates
            type_bits * self.type_weight,    # Type features
        ], dim=1)

        # Adaptive point spacing calculation
        n_clusters = self.adaptive_cluster_num(candidates, type_num)

        # Perform clustering (modified weighted K-Means)
        kmeans = WeightedKMeans(n_clusters=n_clusters)
        cluster_ids = kmeans.fit_predict(features)

        # Spatial distribution optimization
        points = self.spatial_optimize(
            candidates, 
            cluster_ids, 
            self.radius, 
            self.device,
            self.stage
        )
        
        return points
    
    def spatial_optimize(self, points, cluster_ids, min_spacing, device, stage):
        """optimization of spatial distribution
        """
        # select the points according to the cluster id
        centroids = []
        unique_ids = torch.unique(cluster_ids)
        for i in unique_ids:
            cluster_pts = points[cluster_ids == i]
            centroid = cluster_pts.mean(dim=0)
            # make sure that the centroid is travisible
            distances = torch.sum((cluster_pts - centroid) ** 2, dim=1)
            closest_idx = torch.argmin(distances)
            centroid = cluster_pts[closest_idx]
            centroids.append(centroid)
        # distance filter
        if stage not in self.final_pts:
            self.final_pts[stage] = []
        final_pts_new = []
        for pt in sorted(centroids, key=lambda x: -x[0]):
            if all(torch.norm(pt[0] - exist[0]) >= min_spacing 
                   for exist in final_pts_new):
                final_pts_new.append(pt)
                self.final_pts[stage].append(pt)
        
        return torch.stack(final_pts_new) \
            if final_pts_new else torch.empty((0,2), device=device)
    
    def get_point_next_stage(self, bev:torch.Tensor, scene_graph:nx.Graph, bev_wall, oc_map):
        '''if there are points can be matched in next stage
        '''
        # nav tast of next stage
        if self.stage == self.end_stage:
            return None, 0
        nav_mask = torch.zeros_like(bev).to(self.device)
        dag_edge_stage_next = self.instruction_dag[self.stage+1][self.stage+2]
        coordination = []
        if self.stage in self.final_pts:
            for node in self.final_pts[self.stage]:
                coordination.append(node.to(dtype=torch.float))
        if len(coordination) == 0:
            return None, 0
        
        coordination = torch.stack(coordination)
        mean_coordination = torch.mean(coordination, dim=0)
        dis_2 = torch.sum((coordination - mean_coordination)**2, dim=1)
        start_point = torch.tensor(self.navigation_tree.stage_begin[self.stage], device=self.device)
        dis_mean_start = torch.sum((mean_coordination - start_point)**2, dim=0)
        dis_coor_start = torch.sum((coordination - start_point)**2, dim=1)
        dis_2 = torch.where(dis_coor_start<dis_mean_start, dis_2, dis_2+1000)
        mean_coordination = coordination[torch.argmin(dis_2)].to(torch.int)
        dy = mean_coordination[0] - start_point[0]
        dx = mean_coordination[1] - start_point[1]
        angle_offset = torch.atan2(dy, dx)
        angle_agent_t = angle_offset

        for location in coordination:
            nav_constraint = Constraint.set_constraint(
                "navigation_direction",
                dag_edge_stage_next['direction'],
                dag_edge_stage_next['position2next'],
                [location.tolist()],
                dataset=self.dataset
            )
            nav_mask = nav_mask | nav_constraint.draw_mask(bev.shape, angle_agent_t, oc_map)
        for point in self.navigation_tree.object_detect[self.stage+1]:
            x, y = point
            x0 = torch.max(x-self.radius, self.map_min_size).to(torch.long)
            x1 = torch.min(x+self.radius, self.map_max_size).to(torch.long)
            y0 = torch.max(y-self.radius, self.map_min_size).to(torch.long)
            y1 = torch.min(y+self.radius, self.map_max_size).to(torch.long)
            nav_mask[x0:x1, y0:y1] = False

        # object mask union
        mask_list = self.update_instance(self.first_point, scene_graph, self.stage+1)
        angle_agent_tt = angle_agent_t + dag_edge_stage_next['position2next']

        if mask_list is None or len(mask_list) == 0:    
            return None, 0

        constraint_list = []
        mask_draw_list = []
        for _, mask in mask_list.items():
            constraint_list.append(list(range(len(mask['constraint']))))
            mask_draw = []
            for _constraint in mask['constraint']:
                mask_draw.append(
                    _constraint.draw_mask(bev.shape, angle_agent_tt, oc_map)
                )
            mask_draw_list.append(mask_draw)
        
        for combination in itertools.product(*constraint_list):
            objects_mask = torch.ones_like(bev).to(self.device)
            mask_used = []
            mask_names = []
            object_masks = []
            for _index, (key, mask), mask_draw in \
                zip(combination, mask_list.items(), mask_draw_list):
                _mask = mask_draw[_index]
                objects_mask = objects_mask & _mask
                object_masks.append(_mask)
                mask_used.append(key)
                mask_names.append(mask['nodes'][_index])
            if len(mask_names) != len(np.unique(mask_names)):
                continue

            whole_mask = objects_mask & nav_mask & bev
            if torch.sum(whole_mask) > 50:
                # Get all candidate points coordinates
                y, x = torch.where(whole_mask)
                candidates = torch.stack([y, x], dim=1).float()
                # Build composite feature space
                h, w = bev.shape
                coord_features = candidates / torch.tensor([[h, w]], device=self.device)
                kmeans = WeightedKMeans(n_clusters=1)
                cluster_ids = kmeans.fit_predict(coord_features)

                points = self.spatial_optimize(
                    candidates, 
                    cluster_ids, 
                    self.radius, 
                    self.device,
                    self.stage+1
                )
                if len(points) == 0:
                    continue
                next_point = np.array([int(points[0][0]), int(points[0][1])])
                start_point = np.array(self.navigation_tree.stage_begin[self.stage])
                candidate = np.array([[int(mean_coordination[0]), int(mean_coordination[1])],
                                      next_point])
                dis = np.sum((candidate - start_point)**2, axis=1)
                if dis[0] > dis[1]-0.5:
                    next_start_point = start_point
                else:
                    next_start_point = mean_coordination
                # update the graph
                next_start_point_str = \
                    str(int(next_start_point[0]))+','+str(int(next_start_point[1]))
                bev_nt = bev
                next_point_str = str(int(next_point[0]))+','+str(int(next_point[1]))
                next_point_str = self.navigation_tree.update_navigation_tree(
                    bev_nt,
                    [next_point_str], 
                    [mask_used], 
                    [mask_names], 
                    [self.stage+1]
                )
                if next_point_str is None:
                    continue

                self.navigation_tree.path.append(next_start_point_str) 
                next_start_point = [int(item) for item in next_start_point_str.split(',')]
                self.navigation_tree.stage_begin[self.stage+1] = next_start_point

                self.navigation_tree.path.append(next_point_str)
                next_point = np.array([int(item) for item in next_point_str.split(',')])
                self.navigation_tree.stage_begin[self.stage+2] = next_point.tolist()

                self.masks[self.stage+1] = {
                    'object_masks': object_masks,
                    'valid_mask': whole_mask,
                    'nav_mask': whole_mask,
                }

                if self.thin_or_not == 1:
                    self.navigation_tree.waypoints_tree.nodes[next_start_point_str]['visited'] = True
                    self.navigation_tree.navigation_tree.nodes[next_start_point_str]['visited'] = True
                    self.navigation_tree.waypoints_tree.nodes[next_point_str]['visited'] = True
                    self.navigation_tree.navigation_tree.nodes[next_point_str]['visited'] = True

                    # return the time_sequence of the next_point
                    stage_t = self.navigation_tree.waypoints_tree.nodes[next_point_str]['stage']
                    match_result = self.navigation_tree.waypoints_tree.nodes[next_point_str]['match_result']
                    match_sequence = 0 if len(match_result) == 0 else max(match_result) + 1
                    time_sequence = stage_t * 100 + match_sequence
                    return next_point, time_sequence
                else:
                    stage_t = self.stage + 1
                    match_sequence = len(mask_used)
                    time_sequence = stage_t * 100 + match_sequence
                    _point = torch.from_numpy(next_point).to(device=self.device, dtype=torch.int32)
                    self.navigation_tree.object_detect[self.stage+1].append(_point)
                    self.navigation_tree.point_choose[self.stage+1].append(_point)
                    return next_point, time_sequence

        return None, 0
    
    def update_first_point_angle(self, agent_location_t=None):
        if agent_location_t:
            self.first_point = agent_location_t
        else:
            self.first_point = self.navigation_tree.stage_begin[self.stage]
        
        stage_begin_dict = self.navigation_tree.stage_begin
        if self.stage <= 1:
            self.first_angle = 0.
        else:
            point_now = torch.tensor(stage_begin_dict[self.stage])
            point_pre = torch.tensor(stage_begin_dict[self.stage-1])
            self.first_angle = torch.atan2(point_now[0]-point_pre[0], point_now[1]-point_pre[1])

    def random_explore(self, agent_location, bev):
        iteration = 0
        random_list = list(range(-self.random_radius, -5)) + list(range(5, self.random_radius))
        y = int(agent_location[0] + random.choice(random_list))
        x = int(agent_location[1] + random.choice(random_list))
        while bev[y, x] == 0 and iteration < 50:
            iteration += 1
            y = int(agent_location[0] + random.choice(random_list))
            x = int(agent_location[1] + random.choice(random_list))
        
        if iteration < 50:
            return np.array([y, x], dtype=np.int32)
        else:
            return None
        
    def find_nearest_point(self, skeleton_bev):
        '''find an equivalent starting point on the bev_thin graph
        '''
        ys, xs = torch.where(skeleton_bev)
        raw_point = torch.stack([ys, xs], dim=1)
        if len(raw_point) == 0:
            return None
        skeleton_pts = raw_point.float()
        target = torch.tensor([self.map_size//2, self.map_size//2], device=skeleton_pts.device)
        deltas = skeleton_pts - target
        sq_distances = torch.sum(deltas ** 2, dim=1)
        min_idx = torch.argmin(sq_distances)
        nearest_pt = raw_point[min_idx].long().tolist()

        return nearest_pt

    def get_next_point(
            self, 
            agent_location, 
            agent_angle, 
            scene_graph, 
            action_valid, 
            last_round, 
            bev_wall, 
            bev, 
            thin_type, 
            bev_thin,
        ):
        '''get the next point in each turn
        '''
        # update the begin point
        if self.navigation_tree.bp_flag == 0:
            if thin_type == 0:
                self.navigation_tree.update_begin_point(
                    [self.map_size//2, self.map_size//2], bev.shape, self.thin_or_not
                )
                self.thin_or_not = thin_type
            else:
                bp = self.find_nearest_point(bev_thin)
                self.navigation_tree.update_begin_point(bp, bev_thin.shape, self.thin_or_not)
                self.thin_or_not = thin_type

        _new_stage = self.new_stage
        self.new_stage = False
        # x is column, y is row [row, column]
        agent_location_t = [
            agent_location[1] * self.resolution + self.map_size//2, 
            agent_location[0] * self.resolution + self.map_size//2
        ]
        # if action plan failed
        if (action_valid == 'incomplete') and _new_stage:
            if self.stage in self.masks:
                stage_t = self.stage
            else:
                stage_t = self.stage - 1
            next_point, self.stage, stage_end, time_sequence = \
                self.navigation_tree.back_track(stage_t)
            if time_sequence > self.best_sequence:
                self.best_point = next_point
                self.best_sequence = time_sequence
            if self.stage == 0 or next_point is None:
                if self.random_explore_time < self.random_explore_threshold:
                    self.random_explore_time += 1
                    next_point = None
                    if next_point is not None:
                        return next_point, 'continue'
                return self.best_point, 'episode end'
            else:
                self.update_first_point_angle()
                return next_point, 'continue'
        if self.first_point == None:
            self.update_first_point_angle(agent_location_t)

        # update the navigation tree utilizing the current observation
        self.update_instance(self.first_point, scene_graph)

        self.navigation_mode = 'direction'
        if self.dataset == 'r2r':
            while 'end' not in self.navigation_mode:
                if thin_type == 0:
                    self.region_select(bev, self.first_angle, self.first_point, bev, None)
                    points = self.point_set(bev).to(torch.long)
                    bev_nt = bev_wall
                else:
                    self.region_select(bev_thin, self.first_angle, self.first_point, bev)
                    points = self.point_set(bev_thin).to(torch.long)
                    bev_nt = bev_thin

                if self.navigation_mode == 'object':
                    self.navigation_mode = 'object end'
                elif (len(points) == 0):
                    self.navigation_mode = 'object'
                else:
                    self.navigation_mode = 'direction end'
        else:
            if thin_type == 0:
                self.region_select(bev, self.first_angle, self.first_point, bev, None)
                points = self.point_set(bev).to(torch.long)
                bev_nt = bev_wall
            else:
                self.region_select(bev_thin, self.first_angle, self.first_point, bev)
                points = self.point_set(bev_thin).to(torch.long)
                bev_nt = bev_thin

        point_masks = self.masks[self.stage]['type_tensor'][points[:,0], points[:,1]]
        masks_used = []
        masks_names = []
        current_nodes = []
        stages = []
        for point, point_mask in zip(points, point_masks):
            mask_used = []
            mask_names = []
            current_node = str(int(point[0]))+','+str(int(point[1]))
            # object_masks and keys is one-to-one correspondence
            for i in range(len(self.masks[self.stage]['object_masks'])):
                if i >= 63:
                    break
                if point_mask & (1 << i):
                    key = self.masks[self.stage]['keys'][i]
                    mask_used.append(key[0])
                    current_constraint = self.constraints[self.stage][key[0]]
                    mask_names.append(current_constraint['nodes'][key[1]])
            masks_used.append(mask_used)
            masks_names.append(mask_names)
            current_nodes.append(current_node)
            stages.append(self.stage)
        _ = self.navigation_tree.update_navigation_tree(
            bev_nt,
            current_nodes, 
            masks_used, 
            masks_names, 
            stages
        )
        # judge the end of the stage
        direction = self.first_angle+self.instruction_dag[self.stage][self.stage+1]['position2next']
        if self.instruction_dag[self.stage][self.stage+1]['direction'] == 'right':
            direction += np.pi/3
        elif self.instruction_dag[self.stage][self.stage+1]['direction'] == 'left':
            direction -= np.pi/3
        direction = np.array(direction)
        next_point, stage_end, time_sequence = self.navigation_tree.pop_next_point(
            self.stage, 
            direction, 
            bev, 
            self.masks[self.stage]['nav_mask'], 
        )
        if stage_end == 'next turn':
            self.new_stage = True
            self.pre_point = self.first_point
            self.pre_angle = self.first_angle
            self.pre_stage = self.stage
            self.stage += 1
            self.exp_time_this_stage = 0
            self.first_point = None
            self.first_angle = None
        elif stage_end == 'continue':
            # advancedly match
            if (self.exp_time_this_stage >= self.exp_threshold_one_stage or (_new_stage and action_valid == 'threshold')) \
                and ('direction' in self.navigation_mode or len(list(self.instruction_dag.neighbors(self.stage))) < 2):
                advanced_next_point, time_sequence = \
                    self.get_point_next_stage(bev_nt, scene_graph, bev_wall, bev)
                if advanced_next_point is not None:
                    _stage = self.stage + 2
                    if time_sequence > self.best_sequence:
                        self.best_point = advanced_next_point
                        self.best_sequence = time_sequence
                    if _stage > self.end_stage:
                        return advanced_next_point, 'episode end'
                    else:
                        self.pre_point = self.first_point
                        self.pre_angle = self.first_angle
                        self.new_stage = True
                        self.pre_stage = self.stage
                        self.stage = _stage
                        self.first_point = None
                        self.first_angle = None
                        self.exp_time_this_stage = 0
                        return advanced_next_point, 'next turn'
            # normally match
            self.exp_time_this_stage += 1
            if next_point is None:
                next_point, self.stage, stage_end, time_sequence = \
                    self.navigation_tree.back_track(self.stage)
                if self.stage == 0 or next_point is None:
                    if self.random_explore_time < self.random_explore_threshold:
                        if self.random_explore_time < self.random_explore_threshold:
                            self.random_explore_time += 1
                            next_point = None
                            if next_point is not None:
                                return next_point, 'continue'
                        return self.best_point, 'episode end'
                self.update_first_point_angle()    
            if next_point is not None:
                self.pre_point = next_point

        if time_sequence > self.best_sequence:
            self.best_point = next_point
            self.best_sequence = time_sequence

        if last_round or next_point is None:
            return self.best_point, 'episode end'
        return next_point, stage_end
