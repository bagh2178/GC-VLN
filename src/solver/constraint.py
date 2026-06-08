import torch
import numpy as np
from itertools import combinations
from src.solver.math_utils import norm_ang, ang_diff
from collections import deque


class Constraint():
    '''Mathematical expression of prescribed constraints
    '''
    def __init__(self, type, device='cuda', dataset='rxr'):
        ''' 
        use the function "set_constraint" to set the constraint for the DAG
        Args:
        type:           - 1 for definite direction
                        - 2 for undefinite direction
                        - 3 for determining direction through the line connecting an object
                        - 4 for determining direction based on the positions of a few objects
        center:         Center coordinates (x, y)
        inner_radius:   Inner radius
        outer_radius:   Outer radius
        angle_start:    Starting angle (-pi, pi)
        angle_end:      End angle (-pi, pi)
        '''
        self.device = device
        self.mask = None
        self.angle_direction = 0
        self.key = None
        if type == 1:
            self.type = 'direction'
            self.center = (0, 1)
            self.inner_radius = 10
            self.outer_radius = 100
            if dataset == 'rxr':
                self.angle_start = -np.pi/4
                self.angle_end = np.pi/4
            else:
                self.angle_start = -np.pi/6
                self.angle_end = np.pi/6
        if type == 2:
            self.type = 'near'
            self.center = (0, 1)
            self.inner_radius = 10
            self.outer_radius = 35
            self.angle_start = -np.pi
            self.angle_end = np.pi
        if type == 3:
            self.type = 'through'
            self.center = (0, 1)
            self.inner_radius = 10
            self.outer_radius = 40
            self.angle_start = -np.pi
            self.angle_end = np.pi
        if type == 4:
            self.type = 'weave'
            self.center = (0, 1)
            self.inner_radius = 10
            self.outer_radius = 40
            self.angle_start = -np.pi
            self.angle_end = np.pi

    def draw_mask(self, shape, angle_agent, bev):
        """
        Args:
        shape:          the shape of binary image in tensor
        angle_agent:    angle of the agent (-pi, pi) left is positive
        bev:            occupancy map
        Return:               
        the mask of the constraint stroed in self.mask
        """ 
        y, x = torch.meshgrid(
            torch.arange(shape[1], device=self.device),
            torch.arange(shape[0], device=self.device),
        )
        x, y = x - self.center[1], y - self.center[0]
        
        distance = torch.sqrt(x**2 + y**2)
        angle = torch.atan2(y, x)

        if self.type == 'through' or self.type == 'weave':
            angle_offset = 0
        else:
            angle_offset = angle_agent + self.angle_direction
        angle_start_t = norm_ang(self.angle_start + angle_offset)
        angle_end_t = norm_ang(self.angle_end + angle_offset)
        if torch.abs(angle_start_t - angle_end_t) < np.pi/18:
            _mask = (distance <= self.outer_radius)
        elif angle_start_t < angle_end_t:
            _mask = (distance <= self.outer_radius) \
                        & (angle >= angle_start_t) \
                        & (angle <= angle_end_t)
        else:
            _mask = (distance <= self.outer_radius) \
                        & ((angle >= angle_start_t) \
                        | (angle <= angle_end_t))
        self.mask = _mask & (distance >= self.inner_radius)
            
        # fill the mask 
        if self.key == 'relation' and self.type in ['direction', 'near']:
            q = deque([[self.center[0], self.center[1], 0]])
            room_mask = torch.zeros_like(bev, dtype=bool)
            while q:
                _y, _x, _s = q.popleft()
                _s = int(_s) - 0.1
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    ny, nx = int(_y+dy), int(_x+dx)
                    if 0 <= ny < bev.shape[0] and 0 <= nx < bev.shape[1]\
                        and not room_mask[ny, nx] \
                        and bev[ny, nx] >= _s\
                        and _mask[ny, nx]:
                            room_mask[ny, nx] = True
                            q.append([ny, nx, bev[ny, nx]])
            return self.mask & room_mask
            
        return self.mask
        
    @classmethod
    def set_constraint(cls, key, value, angle_direction, args, device='cuda', dataset='rxr'):
        '''
        Args:
        key:   {navigation_diraction} or {relation} in DAG
        value: specific value of key
        args:  necessary parameters for constraint, coordinate represented in tuple (x,y)
        dataset: dataset type ('rxr' or 'r2r')
        '''
        args_t = torch.tensor(args).to(dtype=torch.float32, device=device)
        if key == 'navigation_direction':
            if value in ["front", "right", "left", "back"]:
                Constraint_raw = cls(1, device, dataset)
                if value == 'right':
                    Constraint_raw.angle_start += np.pi / 12
                    Constraint_raw.angle_end += np.pi / 12
                elif value == 'left':
                    Constraint_raw.angle_start -= np.pi / 12
                    Constraint_raw.angle_end -= np.pi / 12
                elif value == 'front':
                    Constraint_raw.outer_radius = 150
            elif value in ["unknown"]:
                Constraint_raw = cls(2, device, dataset)
                if dataset == 'rxr':
                    Constraint_raw.outer_radius = 75
                else:
                    Constraint_raw.outer_radius = 100
            else:
                return None
        elif key == 'relation':
            if value in ["right", "left", "back", "front"]:
                Constraint_raw = cls(1, device, dataset)
                Constraint_raw.outer_radius = 50
                Constraint_raw.angle_start = -np.pi/6
                Constraint_raw.angle_end = np.pi/6
            elif value in ["near", "pass"]:
                Constraint_raw = cls(2, device, dataset)
            elif value in ["through_f"]:
                Constraint_raw = cls(3, device, dataset)
                angle = torch.atan2(args_t[0][0]-args_t[1][0], args_t[0][1]-args_t[1][1])
                Constraint_raw.angle_start = norm_ang(angle - np.pi/2)
                Constraint_raw.angle_end   = norm_ang(angle + np.pi/2)
                Constraint_raw.inner_radius = 5
                Constraint_raw.outer_radius += Constraint_raw.inner_radius
            elif value == "weave":
                # choose the start and end of the maximum angle
                Constraint_raw = cls(4, device, dataset)
                angle = torch.atan2(args_t[0][0]-args_t[1][0], args_t[0][1]-args_t[1][1])
                angles = []
                for i in range(2, len(args_t)):
                    angles.append(torch.atan2(args_t[i][0]-args_t[1][0], args_t[i][1]-args_t[1][1]))
                # there are diff of angs and the angs itself in diff
                diff = torch.tensor([])
                for a,b in combinations(angles, 2):
                    _angle = ang_diff(a, b)
                    if len(diff) == 0:
                        diff = _angle.unsqueeze(0)
                    else:
                        diff = torch.cat((diff, _angle.unsqueeze(0)), dim=0)
                if len(diff) == 0:
                    Constraint_raw.angle_start = norm_ang(angle - np.pi/2)
                    Constraint_raw.angle_end   = norm_ang(angle + np.pi/2)
                else:
                    max_index = torch.argmax(diff[:,0])
                    u = torch.min(diff[max_index][1:])
                    v = torch.max(diff[max_index][1:])
                    if v - u > np.pi:   
                        Constraint_raw.angle_start = v
                        Constraint_raw.angle_end   = u
                    else:
                        Constraint_raw.angle_start = u
                        Constraint_raw.angle_end   = v
            else:
                return None
        elif value != 'through_b':
            raise ValueError('Invalid key')
        
        Constraint_raw.center = args_t[0]
        Constraint_raw.angle_direction = torch.tensor(angle_direction)
        Constraint_raw.key = key
        return Constraint_raw