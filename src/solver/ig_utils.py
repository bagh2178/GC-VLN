import numpy as np
import torch
import networkx as nx
import itertools
import random
from copy import deepcopy
from collections import Counter
from src.solver.math_utils import norm_ang
from collections import deque

class NT_Utils():
    min_match_ratio = 0.50
    def add_node(self, graph, node, parent_nodes, child_nodes, match_situation=None):
        '''add a node to the navigation tree
        '''
        for parent in parent_nodes:
            if not parent in graph:
                raise ValueError('parent node not found')
            
        # delete the reduntant edge
        if len(child_nodes) > 0:
            combinations = list(itertools.product(parent_nodes, child_nodes))
            for combination in combinations:
                if graph.has_edge(*combination):
                    graph.remove_edge(*combination)
        
        add_node_list = []
        if match_situation:
            for i, parent in enumerate(parent_nodes):
                add_node_list.append((parent, node, {'relation':match_situation[i]}))
        else:
            for i, parent in enumerate(parent_nodes):
                add_node_list.append((parent, node))
        for child in child_nodes:
            add_node_list.append((node, child))
        graph.add_edges_from(add_node_list)

    def update_navigation_tree(
            self, 
            bev, 
            current_nodes:list, 
            masks_used, 
            masks_names, 
            stages
        ):
        '''update the navigation tree from skeleton_bev
        '''    
        result = None
        def _bfs(bev, start_point:list, pre):
            nonlocal result
            visited = set()
            queue = deque()
            queue.append([tuple(start_point), pre])
            visited.add(tuple(start_point))

            while queue:
                node, pre_node = queue.popleft()
                r1 = range(max(node[0]-1,0), min(node[0]+2, bev.shape[0]))
                r2 = range(max(node[1]-1,0), min(node[1]+2, bev.shape[1]))
                neigh_8 = [(i, j) for i in r1 for j in r2]
                neigh_8.pop(neigh_8.index(node))
                for neigh in neigh_8:
                    _pre_node = pre_node

                    if not bev[neigh[0], neigh[1]] or neigh in visited:
                        continue

                    visited.add(neigh)
                    current_node = str(int(neigh[0]))+','+str(int(neigh[1]))
                    add_node_list = []
                    remove_node_list = []
                    if current_node in self.navigation_tree:
                        for parent_node in self.waypoints_tree.predecessors(current_node):
                            if self.navigation_tree.nodes[parent_node]['type'] == 'waypoint':
                                if _pre_node != parent_node:
                                    add_node_list.append((_pre_node, current_node))
                                    remove_node_list.append((parent_node, current_node))
                                    break
                        _pre_node = current_node
                        if current_node in current_nodes:
                            if self.navigation_tree.nodes[current_node]['visited'] == False:
                                result = current_node
                    elif current_node in current_nodes:
                        self.existed_point_mask[neigh[0], neigh[1]] = False
                        result = current_node
                        _index = current_nodes.index(current_node)
                        mask_used = masks_used[_index]
                        mask_names = masks_names[_index]
                        stage = stages[_index]
                        # update the object node
                        match_object = np.array(self.match_objects[stage-1])
                        match_result = set()
                        match_situation = []
                        for i, mask in enumerate(mask_used):
                            index = np.where(match_object == mask)[0][0]
                            match_result.add(index)
                            match_situation.append(self.instruction_graph[stage][mask]['relation'])
                            if not mask_names[i] in self.navigation_tree:
                                self.navigation_tree.add_node(mask_names[i], stage=stage, 
                                                            type='object')
                        match_result = list(match_result)
                        match_result.sort(reverse=True)
                        # update the waypoint node
                        _cur_node = [int(item) for item in neigh]
                        _cur_node = torch.tensor(_cur_node, device=self.device, dtype=torch.int32)
                        self.navigation_tree.add_node(current_node, stage=stage, type='waypoint', \
                            match_result=match_result, location=_cur_node, visited=False)
                        self.waypoints_tree.add_node(current_node, stage=stage, type='waypoint', \
                            match_result=match_result, location=_cur_node, visited=False)
                        self.add_node(
                            self.navigation_tree, current_node, mask_names, 
                            [], match_situation
                        )
                        add_node_list.append((_pre_node, current_node))
                        _pre_node = current_node

                    self.waypoints_tree.remove_edges_from(remove_node_list)
                    self.navigation_tree.remove_edges_from(remove_node_list)
                    self.waypoints_tree.add_edges_from(add_node_list)
                    self.navigation_tree.add_edges_from(add_node_list)

                    queue.append([neigh, _pre_node])

        start_point = self.start_point
        _bfs(bev, [start_point[0], start_point[1]], f'{start_point[0]},{start_point[1]}') 

        return result   

    def dfs(self, stage, graph, node):
        '''node should exists in the graph
        '''
        successors = set()
        if self.navigation_tree.nodes[node]['stage'] == stage:
            successors.add(node)
        def dfs_t(stage, graph_t, node_t):
            nonlocal successors
            for neighbor in graph_t.successors(node_t):
                if self.navigation_tree.nodes[neighbor]['stage'] == stage:
                    successors.add(neighbor)
                dfs_t(stage, graph_t, neighbor)
        
        dfs_t(stage, graph, node)
        return successors

    def bfs(self, stage, graph, start_node):
        visited = set()
        queue = deque()
        traversal_order = []

        queue.append(start_node)
        visited.add(start_node)

        while queue:
            current_node = queue.popleft()
            if graph.nodes[current_node]['visited'] == False and \
                graph.nodes[current_node]['stage'] == stage:
                traversal_order.append(current_node)
                break

            for neighbor in graph.successors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return traversal_order

    def pop_next_point(self, stage, direction, bev, mask, thin_type=1):
        '''pop the next node for the navigation in certain turn 
        Args:
        - stage: the stage that the node to be popped is in
        '''
        point_pre = np.array([int(item) for item in self.path[-1].split(',')])
        all_nodes = self.dfs(stage, self.waypoints_tree, self.start_point_str)
        nodes_late_timeline = []
        match_result = {}
        # timeline priority
        for node in list(all_nodes):
            location = self.waypoints_tree.nodes[node]['location']
            if not mask[location[0], location[1]]:
                continue
            if bev[location[0], location[1]] == 0:
                self.waypoints_tree.nodes[node]['visited'] = True
                self.navigation_tree.nodes[node]['visited'] = True
            for index in self.waypoints_tree.nodes[node]['match_result']:
                if index not in match_result:
                    # [name in nx, coordinate]
                    match_result[index] = [node]
                else:
                    match_result[index].append(node)
            if self.waypoints_tree.nodes[node]['visited'] == False:
                nodes_late_timeline.append(node)
        pairs = []
        combinations = []
        for key, value in match_result.items():
            combinations.append(value)
        for combination in itertools.product(*combinations):
            length = len(combination)
            if length > 1:
                if any(node in nodes_late_timeline for node in combination):
                    pairs.append(combination)
            elif length > 0:
                if combination[0] in nodes_late_timeline:
                    pairs.append(combination)
        
        # choose one point
        angles = []
        start_point = np.array(self.stage_begin[stage])
        if thin_type == 1:
            start_point_str = f'{start_point[0]},{start_point[1]}'
        else:
            start_point_str = f'{self.start_point[0]},{self.start_point[1]}'
        if len(pairs) > 0:
            for pair in pairs:
                end_point = np.array(
                    [int(item) for item in pair[-1].split(',')]
                )
                angle = np.arctan2(end_point[0]-start_point[0], end_point[1]-start_point[1])
                angle = norm_ang(angle-direction)
                angles.append(angle)
            angles = np.stack(angles)
            _index = np.argmin(np.abs(angles))
            next_point = np.array(
                [int(item) for item in pairs[_index][-1].split(',')]
            )
            for node in pairs[_index]:
                self.waypoints_tree.nodes[node]['visited'] = True
                self.navigation_tree.nodes[node]['visited'] = True
                for index in self.waypoints_tree.nodes[node]['match_result']:
                    self.matching_sum[stage].add(index)
        else:
            points = self.bfs(stage, self.waypoints_tree, start_point_str)
            if len(points) > 0:
                next_point = np.array(
                    [int(item) for item in points[0].split(',')]
                )
            else:
                next_point = None
                return next_point, 'continue', 0

        next_point_str = f'{next_point[0]},{next_point[1]}'
        if np.any(next_point != point_pre):
            self.path.append(next_point_str)
        self.waypoints_tree.nodes[self.path[-1]]['visited'] = True
        self.navigation_tree.nodes[self.path[-1]]['visited'] = True
        # whether the current stage is satisfied
        if thin_type == 1:
            if (len(match_result) >= len(self.match_objects[stage-1]) * self.min_match_ratio - 0.1) and (
                (stage == self.end_stage) or (len(self.match_objects[stage-1]) != 0) or 
                (len(self.match_objects[stage]) == 0)
            ):
                self.stage_begin[stage+1] = next_point.tolist()
                if stage == self.end_stage:
                    result = 'episode end'
                else:
                    result = 'next turn'
            else:
                result = 'continue'
        elif thin_type == 2:
            if (len(list(self.matching_sum[stage])) >= len(self.match_objects[stage-1]) * self.min_match_ratio - 0.1) and \
                ((len(self.match_objects[stage-1]) != 0) or (len(self.match_objects[stage]) == 0)):
                self.matching_sum[stage].clear()
                self.stage_begin[stage+1] = next_point.tolist()
                if stage == self.end_stage:
                    result = 'episode end'
                else:
                    result = 'next turn'
            else:
                result = 'continue'

        # return the time_sequence of the next_point
        stage_t = self.waypoints_tree.nodes[next_point_str]['stage']
        match_result = self.waypoints_tree.nodes[next_point_str]['match_result']
        match_sequence = 0 if len(match_result) == 0 else max(match_result) + 1
        time_sequence = stage_t * 100 + match_sequence

        _point = torch.from_numpy(next_point).to(device=self.device, dtype=torch.int32)
        self.point_choose[stage].append(_point)
        if len(match_result) > 0:
            self.object_detect[stage].append(_point)
        
        return next_point, result, time_sequence

    def back_track(self, stage):
        '''find the last point in the navigation tree
        '''
        next_node = None
        for stage_goal in range(stage, 0, -1):
            for stage_t in range(stage, 0, -1):
                start_point = np.array(self.stage_begin[stage_t])
                start_point_str = f'{start_point[0]},{start_point[1]}'
                points = self.bfs(stage_goal, self.waypoints_tree, start_point_str)
                if len(points) > 0:
                    next_node = np.array(
                        [int(item) for item in points[0].split(',')]
                    )
                    break
            if next_node is not None:
                break
        if next_node is None:
            return next_node, 1, 'continue', 0
            
        next_node_list = next_node.tolist()
        next_node_str = f'{next_node[0]},{next_node[1]}'
        # correct the path and stage
        self.navigation_tree.nodes[next_node_str]['visited'] = True
        self.waypoints_tree.nodes[next_node_str]['visited'] = True

        stage_t = self.waypoints_tree.nodes[next_node_str]['stage']
        match_result = self.waypoints_tree.nodes[next_node_str]['match_result']
        if (len(match_result) >= len(self.match_objects[stage_t-1]) * self.min_match_ratio - 0.1) and (
            (stage_t == self.end_stage) or (len(self.match_objects[stage_t-1]) != 0) or 
            (len(self.match_objects[stage_t]) == 0)
        ):
            if stage_t == self.end_stage:
                result = 'episode end'
            else:
                result = 'next turn'
                stage_t += 1
        else:
            result = 'continue'

        path_t = deepcopy(self.path)
        for i in range(-1, -len(path_t), -1):
            if not nx.has_path(self.waypoints_tree, path_t[i], next_node_str):
                self.path.pop()
            else:
                break
        
        self.path.append(next_node_str)
        for j in range(stage, stage_t, -1):
            self.stage_begin.pop(j, 'stage not exist')
        if not stage_t in self.stage_begin:
            self.stage_begin[stage_t] = next_node_list

        # return the time_sequence of the next_point
        match_result = self.waypoints_tree.nodes[next_node_str]['match_result']
        match_sequence = 0 if len(match_result) == 0 else max(match_result) + 1
        time_sequence = stage_t * 100 + match_sequence
        return next_node, stage_t, result, time_sequence
