import json
import networkx as nx
import numpy as np

from src.solver.math_utils import norm_ang


def get_instruction_graphs(observations):
    """Convert JSON DAG to NetworkX graph.

    Args:
        observations: List of observations containing instruction['DAG']

    Returns:
        stage_num: Number of stages per instruction
        instruction_graphs: List of NetworkX DiGraph objects
        objects_sum: List of object names to match per instruction
        wrong_dags: List of indices with invalid DAGs (currently unused)
    """
    instruction_graph_dict = []
    for i in range(len(observations)):
        instruction_graph_dict.append(observations[i]['instruction']['DAG'])

    instruction_graphs = []
    stage_num = []
    objects_sum = []
    wrong_dags = []

    for index, dag_str in enumerate(instruction_graph_dict):
        objects_sum_single = []
        list_t = json.loads(dag_str)
        stage_num.append(len(list_t))

        # Create directed graph
        G = nx.DiGraph()
        G.add_node(1, type='waypoint', index=-1)

        for key, value in list_t.items():
            if isinstance(key, str) and 'stage' in key:
                key = key.split('_')[1]
            key = int(key)

            # Convert navigation direction to angle
            directions = value['navigation_direction'].split(' ')
            direction_result = 0
            for direction in directions:
                if direction == 'left':
                    direction_result += 90
                elif direction == 'right':
                    direction_result += 270
                elif direction == 'back':
                    direction_result += 180
            direction_result = direction_result % 360
            direction_result = norm_ang(direction_result / 180.0 * np.pi)

            # Add edge with direction info
            G.add_edge(key, key + 1, position2next=direction_result, direction=directions[0])
            G.nodes[key + 1]['type'] = 'waypoint'
            G.nodes[key + 1]['index'] = len(value['connected_nodes'])

            # Add object nodes
            for obj_index, object in enumerate(value['connected_nodes']):
                if '*' not in object['node']:
                    objects_sum_single.append(object['node'])

                    # Find occurrence count for naming
                    same_name_list = [n for n in G.nodes if str(n).split('_')[0] in object['node']]
                    times = len(same_name_list)

                    if object['relation'] == 'through':
                        G.add_edge(key, object['node']+'_'+str(times), relation='through_f')
                        G.add_edge(object['node']+'_'+str(times), key+1, relation='through_b')
                        G.nodes[object['node']+'_'+str(times)]['type'] = 'object'
                        G.nodes[object['node']+'_'+str(times)]['index'] = obj_index
                    elif object['relation'] == 'weave':
                        names = object['node'].split(',')
                        for i, name in enumerate(names):
                            num = names[:i].count(name)
                            G.add_edge(key, name+'_'+str(times+num), relation='weave')
                            G.nodes[name+'_'+str(times+num)]['type'] = 'object'
                            G.nodes[name+'_'+str(times+num)]['index'] = obj_index
                    else:
                        G.add_edge(key, object['node']+'_'+str(times), relation=object['relation'])
                        G.nodes[object['node']+'_'+str(times)]['type'] = 'object'
                        G.nodes[object['node']+'_'+str(times)]['index'] = obj_index

        instruction_graphs.append(G)
        objects_sum.append(list(set(objects_sum_single)))

    return stage_num, instruction_graphs, objects_sum, wrong_dags