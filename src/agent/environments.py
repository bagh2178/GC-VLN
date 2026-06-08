import cv2
from typing import Any, Dict, Optional, Tuple, List, Union
import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from src.agent.utils import generate_video, navigator_video_frame, planner_video_frame
from scipy.spatial.transform import Rotation as R
from src.agent.navigation_planner import FMMPlanner


def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat

def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist

@baseline_registry.register_env(name="GCVLNEnv")
class GCVLNEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.trans = None
        self.pre_location = None
        self.pre_episode_id = None
        self.gcvln_config = config
        self.prev_episode_id = "something different"

        self.tele_flag = config.POLICY_CONFIG.UNI.TELEFLAG
        self.video_option = config.POLICY_CONFIG.UNI.VIDEO_OPTION
        self.video_dir = config.POLICY_CONFIG.UNI.VIDEO_DIR
        self.map_size = config.POLICY_CONFIG.MAP.SIZE
        self.map_resolution = config.POLICY_CONFIG.MAP.RESOLUTION
        self.observation_map = np.zeros(((self.map_size+1)//20, (self.map_size+1)//20), dtype=np.uint8)
        self.video_frames = []
        self.plan_frames = []

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
    
    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)
    
    def teleport(self, pos, rotation):
        self._env.sim.set_agent_state(pos, rotation)

    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs

    def current_dist_to_goal(self):
        init_state = self._env.sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance
    
    def point_dist_to_goal(self, pos):
        dist = self._env.sim.geodesic_distance(
            pos, self._env.current_episode.goals[0].position,
        )
        return dist
    
    def get_cand_real_pos(self, forward, angle):
        '''get cand real_pos by executing action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_pose = post_state.position

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_pose

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        return circle_dists

    def ghost_dist_to_ref(self, ghost_vp_pos, ref_path):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        # ghost dis to subgoal
        ghost_dists_to_subgoal = []
        for ghost_vp, ghost_pos in ghost_vp_pos:
            dist = self._env.sim.geodesic_distance(ghost_pos, sub_goal_pos)
            ghost_dists_to_subgoal.append(dist)

        oracle_ghost_vp = ghost_vp_pos[np.argmin(ghost_dists_to_subgoal)][0]
        self.prev_episode_id = episode_id
            
        return oracle_ghost_vp

    def get_cand_idx(self, ref_path, angles, distances, candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
            
        return oracle_cand_idx

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def cand_dist_to_subgoal(self, 
        angle: float, forward: float,
        sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def reset(self):
        observations = self._env.reset()
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, 
                    info,
                )
            ]
        return observations
    
    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task 
            )
        return observations

    def get_plan_frame(self, vis_info):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6,6,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
        self.plan_frames.append(frame)

    def trans_tele(self, next_point):
        pos_t = np.array([
            (next_point[1]-self.map_size//2)/self.map_resolution, 
            (next_point[0]-self.map_size//2)/self.map_resolution, 
            1
        ]).reshape(1,3).T
        pos = self.trans @ pos_t
        pos = pos.squeeze()
        pos = np.array([pos[1], self._env.sim.get_agent_state().position[1], pos[0]])
        self.teleport(pos, self._env.sim.get_agent_state().rotation)

    def update_tele_matrix(self):
        self.stuck_time = 0

        p = self._env.sim.get_agent_state().position
        q = self._env.sim.get_agent_state().rotation
        q = [q.x, q.y, q.z, q.w]
        r = R.from_quat(q)
        _, pitch, _ = r.as_euler('xyz', degrees=False)
        pitch = pitch + np.pi
        self.trans = np.array([
            [np.cos(pitch), -np.sin(pitch), p[2]],
            [np.sin(pitch), np.cos(pitch), p[0]],
            [0,0,1]
        ])

    def out_of_map(self, point, bev):
        if point[0] < 0 or point[0] >= self.map_size or point[1] < 0 or point[1] >= self.map_size:
            return True
        elif bev[int(point[0]), int(point[1])] == 0:
            return True
        else:
            return False
        
    def get_gps_location(self):
        agent_state = self._env.sim.get_agent_state()
        # update
        observations = self.get_observation_at(
            agent_state.position, 
            agent_state.rotation
        )
        y, x = \
            (-observations['gps'][1] * self.map_resolution + self.map_size // 2),\
            (observations['gps'][0] * self.map_resolution + self.map_size // 2)
        current_pos = np.array([y, x])
        current_heading = observations['compass'][0]
        return current_pos, current_heading
    
    def step(self, action, vis_info, bev, *args, **kwargs):
        """
        Perform a one-step operation and process visual information 
        based on the type of operation
        
        Args:
            action: dict containing the action to perform
            bev: 2D traversible map, 1 representing traversible
            *args: other positional argument
            **kwargs: other keyword argument
            
        Return:
            observations: the result of observing the environment。
            reward: placeholder for the reward。
            done: whether the episode is done。
            info: additional information
        """
        # parse the operation type
        act = action['act']
        next_point = action['next_point']
        current_pos = action['current_pos']
        current_heading = action['current_heading']
        location_1 = action['location']
        episode_id = self._env.current_episode.episode_id
        action_valid = 'complete'
        if episode_id != self.pre_episode_id:
            self.pre_episode_id = episode_id
            if self.tele_flag:
                self.update_tele_matrix()
            self.observation_map = np.zeros(((self.map_size+1)//20, (self.map_size+1)//20), dtype=np.uint8)
        if next_point is not None:
            # navigation action
            planner = FMMPlanner(
                bev, 
                self.gcvln_config.TASK_CONFIG, 
                self.gcvln_config.POLICY_CONFIG.MAP.RESOLUTION,
            )
            planner.set_goal(next_point)
            result = 'continue'
            while result == 'continue':
                pre_pos = current_pos
                pre_p = self._env.sim.get_agent_state().position
                pre_r = self._env.sim.get_agent_state().rotation
                actions, result = planner.generate_path(
                    current_pos, 
                    current_heading, 
                    act,
                    self.observation_map
                )
                for action_t in actions:
                    self.wrap_act(action_t, vis_info)
                current_pos, current_heading = self.get_gps_location()
                if self.tele_flag:
                    dis = np.sum(np.abs(pre_pos - current_pos))
                    dis_t = np.sum(np.abs(np.array(next_point) - current_pos))
                    if dis <= 3.5 and result != 'incomplete':
                        self.stuck_time += 1
                    elif dis_t >= 150:
                        self.teleport(pre_p, pre_r)
                        result = 'incomplete'
                        break
                    elif dis > 3.5:
                        self.stuck_time = 0
                    if self.stuck_time >= 2 or self.out_of_map(current_pos, bev) \
                        or (result == 'incomplete' and act == 0)\
                        or dis_t > 200:
                        self.trans_tele(next_point)
                        self.stuck_time = 0
                        self.wrap_act(2, vis_info)
                        self.wrap_act(3, vis_info)
                        y = next_point[0]
                        x = next_point[1]
                        result = 'complete'
                        break
            current_pos, current_heading = self.get_gps_location()
            if np.sum(np.abs(pre_pos - current_pos)) > 150 and dis_t <= 200:
                self.teleport(pre_p, pre_r)
                result = 'incomplete'
            action_valid = result
            if result == 'complete':
                self.stuck_time = 0
            y = np.clip(current_pos[0]//20, 0, self.observation_map.shape[0]-1)
            x = np.clip(current_pos[1]//20, 0, self.observation_map.shape[1]-1)
            self.observation_map[int(y), int(x)] = 1
        # navigation action - continue
        if act == 4: 
            # if the video option is enabled, obtain the planning frame
            if self.video_option:
                self.get_plan_frame(vis_info)
    
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(
                agent_state.position, 
                agent_state.rotation
            )

        # navigation action - stop
        elif act == 0:  
            if self.video_option:
                self.get_plan_frame(vis_info)
    
            observations = self._env.step(act)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)
            
        else:
            raise NotImplementedError
    
        # get the reward, done and auxiliary info
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        info['action_valid'] = action_valid
        if next_point is not None:
            info['pre_point'] = str(next_point[0]) + ',' + str(next_point[1])
    
        # if the video option is enabled and the task is done, then generate the video
        if self.video_option and done:
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
        return observations, reward, done, info