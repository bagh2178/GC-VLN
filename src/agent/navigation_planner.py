import torch
import numpy as np
import skfmm
from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt, binary_dilation

class FMMPlanner:
    def __init__(
            self, 
            traversible: torch.tensor,
            config,
            resolution=20, 
            scale=1, 
            step_size=7, 
            robot_radius=0, 
        ):
        """initialize FMMPlanner

        Args
            traversible: accessibility map (1 passable, 0 obstacles)
            config: task config, get the parameter of turning and forward
            resolution: the resolution of the bev map
            scale: map zoom ratio
            step_size: planning step size (in pixels)
            robot_radius: using for expanding the obstacles, pay attention that
            the obstacles have already be expanded when the mask is created
        """
        self.scale = scale
        self.step_size = step_size
        self.turn_angle = config.SIMULATOR.TURN_ANGLE * np.pi / 180
        self.tolerance = 1 / step_size - 0.01
        self.forward_step = config.SIMULATOR.FORWARD_STEP_SIZE * resolution
        self.traversible = traversible.cpu().numpy()
        if scale != 1:
            self.traversible = self._resize_map(traversible)

        # expand obstacles based on the radius of the robot
        if robot_radius > 0:
            self.traversible = self._inflate_obstacles(
                self.traversible, 
                robot_radius
            )

        # FMM distance field
        self.fmm_dist = None
        self.goal = None

        self.max_iterations = 8
        self.iteration = 0
        self.max_distance = None

    def _resize_map(self, traversible:np.ndarray):
        """zoom the map"""

        new_shape = (traversible.shape[0] // self.scale, 
                     traversible.shape[1] // self.scale)
        return resize(traversible, new_shape, mode='constant', anti_aliasing=False) > 0.5

    def _inflate_obstacles(self, traversible, robot_radius):
        """
        expand obstacles based on the radius of the robot

        Args
            traversible: original traversible map
            robot_radius: the radius of the agent(in pixels)
        Return 
            expanded map
        """
        # create the mask of obstacles
        obstacles = (traversible == 0)
        # expand the mask
        structure = np.ones((2 * robot_radius + 1, 2 * robot_radius + 1))
        inflated_obstacles = binary_dilation(obstacles, structure=structure)
        # update the map
        traversible = np.where(inflated_obstacles, 0, 1)
        return traversible

    def set_goal(self, goal):
        """
        set the goal point and calculate the FMM distance field

        Args
            goal: goal point coordination (x, y) in original size without scaling
        """
        # x is row in bev, y is column in bev
        goal_y, goal_x = int(goal[0] / self.scale), int(goal[1] / self.scale)
        # if the goal point is in the obstacles
        if self.traversible[goal_y, goal_x] == 0:
            goal_y, goal_x = self._find_nearest_goal(goal_y, goal_x)
        self.goal = [goal_y / self.scale, goal_x / self.scale]

        traversible_ma = np.ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_y, goal_x] = 0
        self.fmm_dist = skfmm.distance(traversible_ma, dx=1)
        self.fmm_dist = np.ma.filled(self.fmm_dist, np.max(self.fmm_dist) + 1)

    def _find_nearest_goal(self, goal_y, goal_x):
        """find the closest goal point
        """
        max_x, max_y = self.traversible.shape
        # only calculate in the local area
        top_left_selected = (max(0,goal_y-80), max(0,goal_x-80))
        down_right_selected = (min(max_x-1, goal_y+80), min(max_y-1, goal_x+80))
        traversible_temp = np.ones(
            (int(down_right_selected[0]-top_left_selected[0]),
             int(down_right_selected[1]-top_left_selected[1]))
        ) * 1.0
        goal_temp = (goal_y-top_left_selected[0],goal_x-top_left_selected[1])
        traversible_temp[goal_temp[0], goal_temp[1]] = 0
        mask = self.traversible[
            int(top_left_selected[0]):int(down_right_selected[0]), 
            int(top_left_selected[1]):int(down_right_selected[1])
        ]

        euclidean_dist = distance_transform_edt(traversible_temp)
        dist_map = euclidean_dist * mask
        dist_map[dist_map == 0] = dist_map.max() * 2
        # get the coordinate in 2D
        nearest = np.unravel_index(np.argmin(dist_map), dist_map.shape)

        return int(nearest[0]+top_left_selected[0]), int(nearest[1]+top_left_selected[1])
    
    def _compute_fmm_gradient(self):
        """calculate gradient field
        """
        grad_x = np.zeros_like(self.fmm_dist)
        grad_y = np.zeros_like(self.fmm_dist)
        
        # X - column
        grad_x[:, 1:-1] = (self.fmm_dist[:, 2:] - self.fmm_dist[:, :-2]) / 2.0
        
        # Y - row
        grad_y[1:-1, :] = (self.fmm_dist[2:, :] - self.fmm_dist[:-2, :]) / 2.0
    
        return grad_x, grad_y
    
    def _compute_fmm_gradient_v2(self, point):
        '''gradient for one point
        '''
        side_len = 3
        row_l = max(point[0]-side_len, 0)
        row_h = min(point[0]+side_len, self.fmm_dist.shape[0]-1)
        column_l = max(point[1]-side_len, 0)
        colomn_h = min(point[1]+side_len, self.fmm_dist.shape[1]-1)

        point_list = []
        for i in range(column_l, colomn_h):
            point_list.append([row_l, i])
        for i in range(row_l, row_h):
            point_list.append([i, colomn_h])
        for i in range(column_l+1, colomn_h+1):
            point_list.append([row_h, i])
        for i in range(row_l+1, row_h+1):
            point_list.append([i, column_l])
        point_list = np.array(point_list)
        _point = np.array(point)
        mask = np.any(point_list != _point, axis=1)
        point_list = point_list[mask]
        fmm_list = self.fmm_dist[point_list[:,0], point_list[:,1]]

        dis = fmm_list-self.fmm_dist[point[0], point[1]]
        point_diff = point_list -_point
        fmm_diff = np.linalg.norm(point_diff, axis=1)+1e-6
        grad = dis/fmm_diff

        index = np.argmin(grad)
        grad_t = grad[index]
        angle_t = np.arctan2(point_diff[index][0], point_diff[index][1])

        return np.abs(grad_t), angle_t
        
    def generate_path(self, start_state, initial_heading, act, observation_map):
        """
        generate the path from start to goal

        Args
            start_state: starting point coordinate (x, y) in original size
        Return
            path points: list [(x1, y1), (x2, y2), ...]
        """
        actions = []
        result = 'continue'
        # observation distance
        if act == 0:
            self.max_iterations = 50
        current_state = [
            start_state[0] / self.scale, 
            start_state[1] / self.scale
        ]
        row = int(np.clip(current_state[0], 0, self.fmm_dist.shape[0]-1))
        col = int(np.clip(current_state[1], 0, self.fmm_dist.shape[1]-1))

        if self.max_distance is None:
            self.max_distance = self.fmm_dist[row, col]
            dis_line = np.linalg.norm(
                np.array(self.goal, dtype=np.float32) - np.array(current_state, dtype=np.float32)
            )
            if self.max_distance > dis_line * 3:
                return actions, 'incomplete'

        goal_reached = False
        current_distance = self.fmm_dist[row, col]
        # calculate the next point
        if current_distance < self.forward_step:
            goal_reached = True
            result = 'complete'
        else:
            grad, angle = self._compute_fmm_gradient_v2([row, col])
            if grad < self.tolerance:
                result = 'incomplete'
            else:
                # adaptive step size
                i = np.clip(current_distance / self.max_distance, 0, 1)
                step = self.step_size * (0.5 + 0.5 * i)
                next_y = current_state[0] + step * grad * np.sin(angle)
                next_x = current_state[1] + step * grad * np.cos(angle)
                next_state = [next_y * self.scale, next_x * self.scale]
                actions = self.generate_actions(start_state, next_state, initial_heading)
                
                self.iteration += 1
                if self.iteration >= self.max_iterations and self.iteration < 48:
                    y = int(np.clip(next_state[0]//20, 0, observation_map.shape[0]-1))
                    x = int(np.clip(next_state[1]//20, 0, observation_map.shape[1]-1))
                    if observation_map[y, x] == 1:
                        self.max_iterations += 8

        if goal_reached:
            next_state = [self.goal[0], self.goal[1]]
            actions = self.generate_actions(start_state, next_state, initial_heading)
        elif self.iteration >= self.max_iterations:
            # get the max dis
            result = 'threshold'
        return actions, result

    def _smooth_path(self, path):
        """smooth the path
        """
        x = np.array([p[0] for p in path])
        y = np.array([p[1] for p in path])
        t = np.linspace(0, 1, len(path))
        fx = interp1d(t, x, kind='cubic')
        fy = interp1d(t, y, kind='cubic')
        # tnum of points after interpolation, default is twice the original num
        t_new = np.linspace(0, 1, 2 * len(path))
        smooth_path = np.column_stack((fx(t_new), fy(t_new)))
        return smooth_path

    def generate_actions(self, start_state, next_state, initial_heading):
        """
        generate the action list
        """
        actions = []
        current_location = [start_state[0], start_state[1]]
        current_heading = initial_heading

        dy = next_state[0] - current_location[0]
        dx = next_state[1] - current_location[1]
        target_angle = np.arctan2(dy, dx)
        angle_diff = \
            (target_angle - current_heading + np.pi) % (2 * np.pi) - np.pi

        # generate turning actions
        if abs(angle_diff) > 1e-3:
            turn_steps = int((angle_diff+np.sign(angle_diff)*0.25) // self.turn_angle)
            if turn_steps > 0:
                # turn left, left is positive
                for j in range(turn_steps):
                    actions.append(2)
            else:
                # turn right
                for j in range(-1 * turn_steps):
                    actions.append(3)
            angle_diff_real = turn_steps * self.turn_angle
            current_heading += angle_diff_real

        # generate forward actions
        distance = np.hypot(dx, dy)
        steps = 0
        if distance > 1e-3:
            steps = int((distance+2) // self.forward_step)
            for j in range(steps):
                actions.append(1)

        # update the current location
        distance_real = steps * self.forward_step
        current_location[1] += distance_real * np.cos(current_heading)
        current_location[0] += distance_real * np.sin(current_heading)

        return actions