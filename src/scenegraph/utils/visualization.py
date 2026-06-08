import os
import json
import cv2
from PIL import Image
import torch
import numpy as np
import networkx as nx
import supervision as sv


direction_scale = np.array(Image.open("./src/scenegraph/figures/direction_scale.png"))


def visualize_bev_sg(bev:torch.tensor, scene_graph:nx.Graph, index):
    bev = bev.cpu().numpy()
    bev = np.clip(bev, 0, 1)
    bev = (bev * 255).astype(np.uint8)
    color_image = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)

    colors = [np.random.randint(0, 256, size=3).tolist() for _ in range(len(
        scene_graph.nodes()))]
    
    for j, node in enumerate(scene_graph.nodes()):
        x1 = scene_graph.nodes[node]['scope'][0]
        x2 = scene_graph.nodes[node]['scope'][1]
        color_image = cv2.rectangle(
            color_image, 
            (x2[0], x2[1]),
            (x1[0], x1[1]),
            tuple(colors[j]),
            1,
        )


def visualize_mask(mask:torch.tensor, index):
    mask = mask.cpu().numpy()
    mask = np.clip(mask, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    color_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def draw_layers(
        bev_tensor,
        bev_thin,
        valid_mask_tensor,
        other_mask_tensor,
        waypoints,
        edges,
        scene_graph,
        act,
        next_point,
        stage_begin,
        best_point,
        current_node,
        current_toward,
        thin_type,
    ):
    # bev - black and white
    bev_np = bev_tensor.cpu().numpy().astype(np.uint8)
    bev_np_t = bev_np * 255
    bev_img = cv2.cvtColor(bev_np_t, cv2.COLOR_GRAY2BGR)

    # other mask - blue
    blue_light = np.array([255, 120, 0], dtype=np.uint8)
    blue = np.array([255, 0, 0], dtype=np.uint8)
    mask_np = np.zeros_like(bev_np)
    for mask_tensor in other_mask_tensor:
        mask_np = mask_np | mask_tensor.cpu().numpy().astype(np.uint8)
    bev_img[(mask_np == 1) & (bev_np == 1)] = blue_light
    bev_img[(mask_np == 1) & (bev_np == 0)] = blue

    # valid_mask - yellow
    if valid_mask_tensor is not None:
        yellow = np.array([0, 200, 255] , dtype=np.uint8)
        yellow_light = np.array([120, 255, 255], dtype=np.uint8)
        valid_mask_np = valid_mask_tensor.cpu().numpy().astype(np.uint8)
        bev_img[(valid_mask_np == 1) & (mask_np == 0) & (bev_np == 1)] = yellow_light
        bev_img[(valid_mask_np == 1) & (mask_np == 1) & (bev_np == 1)] = yellow

    # bev_thin - green
    bev_thin_np = bev_thin.cpu().numpy().astype(np.uint8)
    bev_img[bev_thin_np == 1] = np.array([0, 255, 0], dtype=np.uint8)

    # waypoints - green
    for waypoint in waypoints:
        if isinstance(waypoint[0], str) and '_' not in waypoint[0]:
            x, y = [int(item) for item in waypoint[0].split(",")]
        else:
            center = scene_graph.nodes[waypoint[0]]['center']
            x, y =  int(center[1]), int(center[0])
        if 0 <= x < bev_img.shape[1] and 0 <= y < bev_img.shape[0]:
            if waypoint[1]['type'] == 'waypoint':
                cv2.circle(bev_img, (y, x), 3, (0, 255, 0), -1)
            else:
                cv2.circle(bev_img, (y, x), 4, (255,0,255), -1)

    # edges between waypoints - green
    if thin_type == 0:
        for edge in edges:
            if isinstance(edge[0], str) and '_' not in edge[0]:
                x1, y1 = [int(item) for item in edge[0].split(",")]
            else:
                center = scene_graph.nodes[edge[0]]['center']
                x1, y1 =  int(center[1]), int(center[0])
            if isinstance(edge[1], str) and '_' not in edge[1]:        
                x2, y2 = [int(item) for item in edge[1].split(",")]
            else:
                center = scene_graph.nodes[edge[1]]['center']
                x2, y2 =  int(center[1]), int(center[0])
            if 0 <= x1 < bev_img.shape[1] and 0 <= y1 < bev_img.shape[0] \
                and 0 <= x2 < bev_img.shape[1] and 0 <=y2 < bev_img.shape[0]:
                cv2.line(bev_img, (y1, x1), (y2, x2), (0, 255, 0), 1)

    # stage_begin - cyan
    for data in stage_begin.values():
        if isinstance(data[0], str):
            x, y = [int(item) for item in data[0].split(",")]
        else:
            x, y =  data[0], data[1]
        if 0 <= x < bev_img.shape[1] and 0 <= y < bev_img.shape[0]:
            cv2.circle(bev_img, (y, x), 5, (0, 64, 128), -1)

    # best node - orange
    if best_point is not None:
        if isinstance(best_point, str):
            x, y = [int(item) for item in best_point.split(",")]
        else:
            x, y =  int(best_point[0]), int(best_point[1])
        if 0 <= x < bev_img.shape[1] and 0 <= y < bev_img.shape[0]:
            cv2.circle(bev_img, (y, x), 3, (0, 140, 255), -1)
            cv2.circle(bev_img, (y, x), 8, (0, 140, 255), 2)

    # next node - red
    nn = next_point
    if nn is not None:
        if isinstance(nn, str):
            x, y = [int(item) for item in nn.split(",")]
        else:
            x, y =  int(nn[0]), int(nn[1])
        if 0 <= x < bev_img.shape[1] and 0 <= y < bev_img.shape[0]:
            if act == 0:
                cv2.circle(bev_img, (y, x), 3, (0, 0, 255), -1)
                cv2.circle(bev_img, (y, x), 8, (0, 0, 255), 2)
            else:
                cv2.circle(bev_img, (y, x), 5, (0, 0, 255), -1)

    # current toward - purple
    if current_node is not None and current_toward is not None:
        if isinstance(current_node, str):
            x1, y1 = [int(item) for item in current_node.split(",")]
        else:
            x1, y1 =  int(current_node[0]), int(current_node[1])
        x2 = int(x1 + 20 * np.sin(current_toward))
        y2 = int(y1 + 20 * np.cos(current_toward))
        if 0 <= x1 < bev_img.shape[1] and 0 <= y1 < bev_img.shape[0] \
            and 0 <= x2 < bev_img.shape[1] and 0 <= y2 < bev_img.shape[0]:
            cv2.arrowedLine(
                bev_img, (y1, x1), (y2, x2), (128, 0, 128), 
                thickness=3, line_type=cv2.LINE_AA, tipLength=0.3
            )

    return bev_img

def draw_panorama(panorama, input_boxes, masks, labels, confidences):
    img = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
    class_ids = np.array(list(range(len(labels))))
    confidences_t = confidences
    labels_t = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(labels, confidences_t)
    ]

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_t)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame


def combine_image(observations, stat_eps, episode_id, stage, bev_img, annotated_frame):
    visualization_image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    visualization_image = add_resized_image(visualization_image, annotated_frame, x1=127, y1=20, size=(1026, 240))
    visualization_image = add_resized_image(visualization_image, direction_scale, x1=99, y1=265, size=(int(1026 * 2970 / (2970 - 2 * 78)) - 4, 30))
    visualization_image = add_resized_image(visualization_image, bev_img, x1=60, y1=300, size=(400, 400))
    visualization_image = add_rectangle(visualization_image, x1=127, x2=1152, y1=20, y2=260, color=(128, 128, 128), thickness=1)
    visualization_image = add_rectangle(visualization_image, x1=60, x2=460, y1=300, y2=700, color=(128, 128, 128), thickness=1)

    instruction_text_English = observations['instruction']['text_English']
    instruction_DAG = observations['instruction']['DAG']
    parsed_instruction_DAG = json.loads(instruction_DAG)
    instruction_DAG_string = []
    highlight_line_index = []
    for i, (key, value) in enumerate(parsed_instruction_DAG.items()):
        navigation_direction = value['navigation_direction'].ljust(7)
        connected_nodes = value['connected_nodes']
        for node in connected_nodes:
            if 'node_singular' in node:
                del node['node_singular']
        connected_nodes = str(connected_nodes)
        connected_nodes = connected_nodes.replace("', '", "': '").replace("'", '').replace('node: ', '').replace('relation: ', '')
        visualization_text = key + ' ' + navigation_direction + ' ' + connected_nodes
        visualization_text = line_list(visualization_text, 150)
        if i == stage - 1:
            highlight_line_index = [len(instruction_DAG_string) + j for j in range(len(visualization_text))]
        instruction_DAG_string.extend(visualization_text)

    visualization_image = add_text_list(visualization_image, line_list(instruction_text_English, 150), (500, 310), font_scale=0.3, thickness=1)
    visualization_image = add_text_list(visualization_image, instruction_DAG_string, (500, 400), font_scale=0.3, thickness=1, highlight_line_index=highlight_line_index)
    if episode_id in stat_eps:
        metric = stat_eps[episode_id]
        visualization_image = add_text_list(visualization_image, line_list(str(metric), 150), (500, 600), font_scale=0.3, thickness=1)

    return visualization_image

def save_video(save_dir, visualization_image_dict):
    for episode_id, visualization_image_list in visualization_image_dict.items():
        if len(visualization_image_list) == 0:
            continue
        save_video_dir = os.path.join(save_dir, 'video')
        save_video_path = f'{save_video_dir}/vid_{int(episode_id):06d}.mp4'
        os.makedirs(save_video_dir, exist_ok=True)
        height, width, layers = visualization_image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename=save_video_path, fourcc=fourcc, fps=1.0, frameSize=(width, height))
        for visualization_image in visualization_image_list:  
            video.write(visualization_image)
        video.release()


def add_resized_image(base_image: np.ndarray, overlay_image: np.ndarray, x1: int, y1: int, size: tuple):
    resized_overlay = cv2.resize(overlay_image, size)

    h, w, c = resized_overlay.shape
    if c == 4:
        resized_overlay = resized_overlay[:, :, :3]

    x, y = int(x1), int(y1)

    if x + w > base_image.shape[1] or y + h > base_image.shape[0]:
        raise ValueError("Overlay image goes out of the bounds of the base image.")

    base_image[y:y+h, x:x+w] = resized_overlay
    return base_image


def line_list(text, line_length=80):
    text_list = []
    for i in range(0, len(text), line_length):
        text_list.append(text[i:(i + line_length)])
    return text_list


def add_text(image: np.ndarray, text: str, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def add_text_list(image: np.ndarray, text_list: list, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 0), thickness=2, highlight_line_index=[]):
    highlight_color = (0, 0, 0)
    not_highlight_color = (128, 128, 128)
    for i, text in enumerate(text_list):
        position_i = (position[0], position[1] + i * 15)
        color = highlight_color if len(highlight_line_index) == 0 or i in highlight_line_index else not_highlight_color
        cv2.putText(image, text, position_i, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def add_rectangle(image: np.ndarray, x1: int, x2: int, y1: int, y2: int, color=(128, 128, 128), thickness=2):
    top_left = (int(x1), int(y1))
    bottom_right = (int(x2), int(y2))
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image