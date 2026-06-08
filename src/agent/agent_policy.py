import gzip
import json
import os
import warnings
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.distributed as distr
import tqdm

from src.solver.instruction_graph import get_instruction_graphs
from src.solver.region_solver import Region_Solver
from habitat import logger
from src.habitat_extensions import Simulator
from src.agent.panorama_utils import rgbs_to_panorama, rotate_180
from src.scenegraph.scene_graph import SceneGraph
from src.scenegraph.utils.spacial_utils import get_camera_matrix, get_pose_matrix
from src.scenegraph.utils.visualization import (
    combine_image,
    draw_layers,
    draw_panorama,
    save_video,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def gather_list_and_concat(list_of_nums, world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    distr.all_gather(gather_t, tensor)
    return gather_t


class Agent():
    def __init__(self, env, config):
        self.envs = env
        self.config = config
        self.max_len = int(config.POLICY_CONFIG.UNI.MAX_RRAJ_LEN)
        self.device = config.POLICY_CONFIG.UNI.DEVICE
        self.world_size = config.GPU_NUMBERS
        self.local_rank = config.local_rank
        self.thin_type = config.POLICY_CONFIG.UNI.THIN_TYPE
        self.dataset = config.DATASET.DATASET_TYPE.lower()

        self.region_solver = Region_Solver(
            config.POLICY_CONFIG,
            self.device,
            self.dataset
        )
        self.camera_matrix = get_camera_matrix(640, 480, 90)

        self.split = config.TASK_CONFIG.DATASET.SPLIT
        if config.TASK_CONFIG.TASK.TASK_TYPE == 'rxr':
            self.gt_data = {}
            try:
                for role in config.TASK_CONFIG.DATASET.ROLES:
                    with gzip.open(
                        config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                            split=self.split, role=role
                        ), "rt") as f:
                        self.gt_data.update(json.load(f))
            except:
                self.gt_data = None

        self.sg_list = []
        self.rs_list = []

        self.experiment_id = config.DATASET.EXPERIMENT_ID
        self.visualization_save_dir = os.path.join('./outputs/visualization', config.DATASET.DATASET_TYPE, self.split, self.experiment_id)
        self.log_save_dir = os.path.join('./outputs/logs', config.DATASET.DATASET_TYPE, self.split, self.experiment_id)
        os.makedirs(self.visualization_save_dir, exist_ok=True)
        os.makedirs(self.log_save_dir, exist_ok=True)

    def _reset_policy(
        self,
        observations,
        instruction_graphs,
        end_stages,
        objects_sum,
        wrong_dags,
    ):
        if len(self.sg_list) == 0:
            for i in range(len(observations)):
                # scenegraph - agent observation
                self.sg_list.append(SceneGraph(self.config.POLICY_CONFIG, self.dataset))
                # region-solver - agent decision-making
                self.rs_list.append(deepcopy(self.region_solver))

        for i in range(len(observations)):
            if i not in wrong_dags:
                self.sg_list[i].reset(dag=instruction_graphs[i])
                self.rs_list[i].update_info(instruction_graphs[i], end_stages[i])

    @staticmethod
    def _pause_envs(envs, envs_to_pause, observations):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                observations.pop(idx)
                envs.pause_at(idx)

        return envs, observations

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def calculate_metric(self, info):
        '''calculate the metric for the finished episode
        '''

        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = info['distance_to_goal']
        metric['success'] = info['success']
        metric['oracle_success'] = info['oracle_success']
        metric['path_length'] = info['path_length']
        metric['spl'] = info['spl']

        return metric

    def rollout(self):
        # resume all the envs
        self.envs.resume_all()
        # reset the envs and get the observations
        observations = self.envs.reset()
        # get the instruction DAG and the region solver
        stage_num, instruction_graphs, objects_sum, wrong_dags = \
            get_instruction_graphs(observations)
        # update the info using for the policy in each episode
        self._reset_policy(
            observations, 
            instruction_graphs, 
            stage_num,
            objects_sum,
            wrong_dags
        )
        
        # suspend the evaluated environment
        env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                        if ep.episode_id in self.stat_eps]    
        self.envs, observations = self._pause_envs(self.envs, env_to_pause, observations)
        if self.envs.num_envs == 0: return

        not_done_index = list(range(self.envs.num_envs))

        curr_eps = self.envs.current_episodes()
        self.visualization_image_dict = {curr_eps[i].episode_id: [] for i in range(self.envs.num_envs)}
        # complete - get the point, incomplete - stuck, threshold - get the max dis of one step
        infos = [{'action_valid': 'complete'} for _ in range(self.envs.num_envs)]
        for stepk in range(self.max_len): 
            env_actions = []       
            for i in range(self.envs.num_envs):
                try:
                    # if bev is wrong
                    if not_done_index[i] in wrong_dags:
                        env_actions.append(
                            {
                                'action': {
                                    'act': 0,
                                    'next_point': None,
                                    'current_pos': observations[i]['gps'],
                                    'current_heading': observations[i]['compass'],
                                    'location': [0,0]
                                },
                                'vis_info': None,
                                'bev': None
                            }
                        )
                        continue
                    # get the next point using region solver
                    if infos[i]['action_valid'] == 'threshold':
                        pre_point = infos[i]['pre_point']
                        nav_tree = self.rs_list[not_done_index[i]].navigation_tree
                        flag = 0
                        if self.rs_list[not_done_index[i]].new_stage:
                            flag = self.rs_list[not_done_index[i]].stage - self.rs_list[not_done_index[i]].pre_stage
                            self.rs_list[not_done_index[i]].stage = self.rs_list[not_done_index[i]].pre_stage
                            self.rs_list[not_done_index[i]].first_point = self.rs_list[not_done_index[i]].pre_point
                            self.rs_list[not_done_index[i]].first_angle = self.rs_list[not_done_index[i]].pre_angle
                        if self.thin_type < 2:
                            if pre_point in nav_tree.navigation_tree.nodes:
                                nav_tree.navigation_tree.nodes[pre_point]['visited'] = False
                                nav_tree.waypoints_tree.nodes[pre_point]['visited'] = False
                                while pre_point in nav_tree.path:
                                    index = nav_tree.path.index(pre_point)
                                    nav_tree.path.pop(index)
                        elif self.thin_type == 2:
                            if flag == 2:
                                _s = self.rs_list[not_done_index[i]].stage + 1
                            else:
                                _s = self.rs_list[not_done_index[i]].stage
                            pre_point = np.array([int(item) for item in pre_point.split(',')])
                            pre_point = torch.from_numpy(pre_point).to(device=self.device, dtype=torch.int32)
                            object_detect = self.rs_list[not_done_index[i]].navigation_tree.object_detect[_s]
                            point_choose = self.rs_list[not_done_index[i]].navigation_tree.point_choose[_s]
                            for i in range(len(object_detect) - 1, -1, -1):
                                t = object_detect[i]
                                if torch.all(pre_point == t):
                                    object_detect.pop(i)
                            for i in range(len(point_choose) - 1, -1, -1):
                                t = point_choose[i]
                                if torch.all(pre_point == t):
                                    point_choose.pop(i)
                        
                    # update the bev map and scene graph
                    pose_matrix = get_pose_matrix(observations[i])
                    panoramas = rgbs_to_panorama(observations[i])
                    camera_matrix = self.camera_matrix
                    panoramas, camera_matrix, pose_matrix = rotate_180(panoramas, camera_matrix, pose_matrix)

                    sg_result = self.sg_list[not_done_index[i]].get_scenegraph(
                        self.rs_list[not_done_index[i]].stage, 
                        panoramas[0], 
                        panoramas[1], 
                        camera_matrix, 
                        pose_matrix,
                        [observations[i]['gps'][0], -observations[i]['gps'][1]],
                        self.thin_type, 
                    )
                    bev = self.sg_list[not_done_index[i]].map_draft.bev_map
                    bev_fmm = self.sg_list[not_done_index[i]].map_draft.bev_map_fmm
                    bev_wall = self.sg_list[not_done_index[i]].map_draft.bev_wall
                    bev_thin = self.sg_list[not_done_index[i]].map_draft.bev_thin
                    scene_graph = self.sg_list[not_done_index[i]].map_draft.scene_graph

                    # the location of the agent will adapt to bev in the function
                    next_point, stage_result = self.rs_list[not_done_index[i]].get_next_point(
                        [observations[i]['gps'][0], -observations[i]['gps'][1]], 
                        observations[i]['compass'][0], 
                        scene_graph, 
                        infos[i]['action_valid'], 
                        stepk == self.max_len - 1, 
                        bev_wall, 
                        bev,
                        self.thin_type,  
                        bev_thin, 
                    )
                    agent_location = np.array([
                        -observations[i]['gps'][1] * self.config.POLICY_CONFIG.MAP.RESOLUTION
                          + self.config.POLICY_CONFIG.MAP.SIZE // 2,
                        observations[i]['gps'][0] * self.config.POLICY_CONFIG.MAP.RESOLUTION
                          + self.config.POLICY_CONFIG.MAP.SIZE // 2
                    ])

                    rs = self.rs_list[not_done_index[i]]
                    vis_info = {
                        'nodes': list(rs.navigation_tree.stage_begin.values()),
                        'ghosts': list(rs.navigation_tree.waypoints_tree.nodes()),
                        'predict_ghost': next_point,
                    }
                    if stage_result == 'episode end' \
                        or stepk == self.max_len - 1 \
                        or next_point is None:
                        # get to the next_point and stop
                        act = 0
                        if next_point is None:
                            stage_begin = rs.navigation_tree.stage_begin
                            next_point = stage_begin[len(stage_begin)-1]
                    else:
                        act = 4
                    # add action to the action list
                    env_actions.append(
                        {
                            'action': {
                                'act': act,
                                'next_point': next_point,
                                'current_pos': agent_location,
                                'current_heading': observations[i]['compass'][0],
                                'location': observations[i]['gps'], 
                            },
                            'vis_info': vis_info,
                            'bev': bev_fmm
                        }
                    )

                    # visualize
                    _stage = self.rs_list[not_done_index[i]].stage
                    if stage_result == 'next turn' or \
                        (stage_result == 'episode end' and stepk == self.max_len - 1):
                        _stage -= 1
                    if _stage == 0:
                        _stage = 1
                    if sg_result:
                        annotated_frame = draw_panorama(
                            panoramas[0],
                            self.sg_list[not_done_index[i]].segment2d_results[-1]['xyxy'],
                            self.sg_list[not_done_index[i]].segment2d_results[-1]['mask'],
                            self.sg_list[not_done_index[i]].segment2d_results[-1]['caption'],
                            self.sg_list[not_done_index[i]].segment2d_results[-1]['caption_conf'],
                        )
                    else:
                        annotated_frame = cv2.cvtColor(panoramas[0], cv2.COLOR_RGB2BGR)
                    other_masks = []
                    for stage_t, value in rs.masks.items():
                        if stage_t == _stage:
                            for j in range(len(value['object_masks'])):
                                if 'keys' in value:
                                    key = value['keys'][j]
                                    current_constraint = rs.constraints[stage_t][key[0]]
                                    mask_name = current_constraint['nodes'][key[1]]
                                    if mask_name in rs.navigation_tree.navigation_tree:
                                        other_masks = other_masks + [value['object_masks'][j]]
                                else:
                                    other_masks = other_masks + [value['object_masks'][j]]
                    if _stage in rs.masks:
                        valid_mask = rs.masks[_stage]['nav_mask']
                    else:
                        valid_mask = None
                    bev_for_vis = bev_fmm if self.dataset == 'rxr' else bev
                    bev_img = draw_layers(
                        bev_for_vis,
                        bev_thin,
                        valid_mask,
                        other_masks,
                        list(rs.navigation_tree.navigation_tree.nodes(data=True)),
                        list(rs.navigation_tree.navigation_tree.edges()),
                        scene_graph,
                        act,
                        next_point,
                        rs.navigation_tree.stage_begin,
                        rs.best_point,
                        agent_location,
                        observations[i]['compass'][0],
                        self.thin_type,
                    )

                    curr_eps = self.envs.current_episodes()
                    visualization_image = combine_image(observations[i], self.stat_eps, curr_eps[i].episode_id, _stage, bev_img, annotated_frame)
                    self.visualization_image_dict[curr_eps[i].episode_id].append(visualization_image)

                except KeyboardInterrupt:
                    print("KeyboardInterrupt received. Exiting...")
                    exit(1)
                except Exception as e:
                    action = {
                            'action': {
                                'act': 0,
                                'next_point': None,
                                'current_pos': observations[i]['gps'],
                                'current_heading': observations[i]['compass'],
                                'location': [0,0]
                            },
                            'vis_info': None,
                            'bev': None
                        }
                    if len(env_actions) > i:
                        env_actions[i] = action
                    else:
                        env_actions.append(action)

            # execute the action in corresponding envs
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate the metric for the finished episode
            for k in range(self.envs.num_envs):
                curr_eps = self.envs.current_episodes()
                ep_id = curr_eps[k].episode_id
                metric = self.calculate_metric(infos[k])
                self.stat_eps[ep_id] = metric
                if not dones[k]:
                    continue
                self.pbar.update()

            # pause completed environment
            if sum(dones) > 0:
                for m in reversed(list(range(self.envs.num_envs))):
                    if dones[m]:
                        not_done_index.pop(m)
                        self.envs.pause_at(m)
                        observations.pop(m)
                        infos.pop(m)

            if self.envs.num_envs == 0:
                break

        # write the result of each episode
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.log_save_dir,
            f"zero_shot_vln_{split}_r{self.local_rank}_w{self.world_size}_{self.config.DATASET.SPLIT_INDEX}({self.config.DATASET.SPLIT_NUM}).json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)
        save_video(self.visualization_save_dir, self.visualization_image_dict)

    def act(self):
        '''experiment main function
        '''
        # experiment
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval)
        while len(self.stat_eps) < eps_to_eval:
            self.rollout()
        self.envs.close()

        # log
        if self.world_size > 1:
            distr.barrier()

        for ep_id, metric in self.stat_eps.items():
            if not 0 <= metric['spl'] <= 1:
                metric['spl'] = 0
            if metric['distance_to_goal'] == float('inf'):
                metric['distance_to_goal'] = 10

        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k, v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v, self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        # write the result of each episode
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.log_save_dir,
            f"zero_shot_vln_{split}_r{self.local_rank}_w{self.world_size}_{self.config.DATASET.SPLIT_INDEX}({self.config.DATASET.SPLIT_NUM}).json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)
        # write the aggregated results
        if self.local_rank < 1:
            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")