import networkx as nx
import numpy as np
import torch
from src.solver.ig_utils import NT_Utils


class NavigationTree(NT_Utils):
    '''DAG to record the navigation process
    '''
    def __init__(
            self,
            instruction_graph,
            end_stage,
            min_distance,
            device='cuda'
        ):
        self.device = device
        self.min_distance = min_distance
        self.reset(instruction_graph, end_stage)

    def reset(self, instruction_graph, end_stage):
        self.end_stage = end_stage
        self.matching_sum = {}
        self.object_detect = {}
        self.point_choose = {}
        self.instruction_graph = instruction_graph
        self.navigation_tree = nx.DiGraph()
        self.waypoints_tree = nx.DiGraph()
        self.sequence_tree, self.match_objects = self.get_sequence_tree(instruction_graph)
        self.bp_flag = 0
        self.path = []
        self.stage_begin = {}

    def update_begin_point(self, start_point, bev_shape, thin_type):
        '''must be used on the first entry into get_next_point function
        '''
        if start_point:
            _start_point = start_point
        else:
            _start_point = [int(bev_shape[0]/2), int(bev_shape[1]/2)]
        self.start_point = _start_point
        self.initialize_navigation_tree()
        self.start_point_str = f'{_start_point[0]},{_start_point[1]}'
        if thin_type != 2:
            self.existed_point_mask = torch.ones(bev_shape, device=self.device, dtype=torch.bool)
            self.existed_point_mask[_start_point[0], _start_point[1]] = False
        self.bp_flag = 1
        if len(self.path) > 0:
            self.path[0] = f'{_start_point[0]},{_start_point[1]}'
        else:
            self.path = [f'{_start_point[0]},{_start_point[1]}']
        self.stage_begin[1] = [_start_point[0],_start_point[1]]

    def initialize_navigation_tree(self):
        self.navigation_tree.add_node(
            f'{self.start_point[0]},{self.start_point[1]}',
            stage=1,
            type='waypoint',
            match_result=[],
            location = torch.tensor(
                [self.start_point[0],self.start_point[1]], device=self.device, dtype=torch.int32
            ),
            visited=True
        )
        self.waypoints_tree.add_node(
            f'{self.start_point[0]},{self.start_point[1]}',
            stage=1,
            type='waypoint',
            match_result=[],
            location = torch.tensor(
                [self.start_point[0],self.start_point[1]], device=self.device, dtype=torch.int32
            ),
            visited=True
        )

    def topological_sort(self, graph: nx.DiGraph):
        '''custom_topological_sort for the DAG
        '''
        in_degree = dict(graph.in_degree())
        zero_in_degree = [node for node in graph.nodes() if in_degree[node] == 0]
        # ascending order
        zero_in_degree.sort(key=lambda x: graph.nodes[x]['index'])
        result = []
        while zero_in_degree:
            node = zero_in_degree.pop(0)
            result.append(node)
            for neighbor in graph.successors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)
                    zero_in_degree.sort(key=lambda x: graph.nodes[x]['index'])
        return result

    def get_sequence_tree(self, instruction_graph):
        # objects need to be matched in each stage
        match_objects = []
        match_object = []
        # Get the topological sort of the graph
        sorted_nodes = list(self.topological_sort(instruction_graph))
        # Create a new directed graph for the navigation tree
        sequence_tree = nx.DiGraph()
        # Add nodes and edges to the navigation tree according to the topological order
        stage = 0
        for j in range(len(sorted_nodes)-1):
            u = sorted_nodes[j]
            v = sorted_nodes[j + 1]
            sequence_tree.add_edge(u, v)
            if isinstance(u, int) or (isinstance(u, str) and u.isdigit()):
                stage = int(u)
            sequence_tree.nodes[u]['stage'] = stage
            sequence_tree.nodes[v]['stage'] = stage
            self.matching_sum[stage] = set()
            self.object_detect[stage] = []
            self.point_choose[stage] = []
            if isinstance(v, int) or (isinstance(v, str) and v.isdigit()):
                match_objects.append(match_object)
                match_object = []
            else:
                match_object.append(v)
        match_objects.append(match_object)
        return sequence_tree, match_objects
