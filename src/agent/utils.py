import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import habitat_sim
import numpy as np
import quaternion
import torch
from habitat.core.simulator import Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_to_list,
)
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray
from torch import Tensor

import src.habitat_extensions.maps as maps
from habitat_baselines.common.baseline_registry import baseline_registry

cv2 = try_cv2_import()
obs_trans_to_eq = baseline_registry.get_obs_transformer("CubeMap2Equirect")
UUIDS_EQ = ['rgbback', 'rgbdown', 'rgbfront', 'rgbright', 'rgbleft', 'rgbup']
CUBE2EQ = obs_trans_to_eq(UUIDS_EQ, (224,448))


def observations_to_image(
    observation: Dict[str, Any], info: Dict[str, Any]
) -> ndarray:
    """Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    if "rgb" in observation and len(observation["rgb"].shape) == 4:
        return pano_observations_to_image(observation, info)
    elif "depth" in observation and len(observation["depth"].shape) == 4:
        return pano_observations_to_image(observation, info)

    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame


def pano_observations_to_image(
    observation: Dict[str, Any], info: Dict[str, Any]
) -> ndarray:
    """Creates a rudimentary frame for a panoramic observation. Includes RGB,
    depth, and a top-down map.
    """
    pano_frame = []
    channels = None
    rgb = None
    if "rgb" in observation:
        cnt = observation["rgb"].shape[0]
        rgb = observation["rgb"][
            [*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :
        ]
        channels = rgb.shape[3]
        vert_bar = np.ones((rgb.shape[1], 20, channels)) * 255
        rgb_frame = [rgb[0]]
        for i in range(1, rgb.shape[0]):
            rgb_frame.append(vert_bar)
            rgb_frame.append(rgb[i])
        pano_frame.append(np.concatenate(rgb_frame, axis=1))

    if "depth" in observation:
        cnt = observation["depth"].shape[0]
        observation["depth"] = observation["depth"][
            [*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :
        ]
        if len(pano_frame) > 0:
            assert observation["depth"].shape[0] == rgb.shape[0]
            pano_frame.append(
                np.ones((20, pano_frame[0].shape[1], channels)) * 255
            )
            observation_size = rgb.shape[1:3]
        else:
            observation_size = observation["depth"].shape[1:3]

        vert_bar = np.ones((observation_size[0], 20, 3)) * 255

        depth = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth = np.stack([depth for _ in range(3)], axis=3)

        depth_frame = [
            cv2.resize(
                depth[0], dsize=observation_size, interpolation=cv2.INTER_CUBIC
            )
        ]
        for i in range(1, depth.shape[0]):
            depth_frame.append(vert_bar)
            depth_frame.append(
                cv2.resize(
                    depth[i],
                    dsize=observation_size,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
        pano_frame.append(np.concatenate(depth_frame, axis=1))

    pano_frame = np.concatenate(pano_frame, axis=0)

    if "top_down_map_vlnce" in info:
        k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        k = "top_down_map"
    else:
        k = None

    if k is not None:
        top_down_map = info[k]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info[k]["fog_of_war_mask"]
        )
        map_agent_pos = info[k]["agent_map_coord"]
        top_down_map = habitat_maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info[k]["agent_angle"],
            agent_radius_px=min(top_down_map.shape[0:2]) // 24,
        )
        if top_down_map.shape[1] < top_down_map.shape[0]:
            top_down_map = np.rot90(top_down_map, 1)

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_width = pano_frame.shape[1] // 3
        top_down_height = int(top_down_width / old_w * old_h)

        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        white = (
            np.ones((top_down_height, pano_frame.shape[1] - top_down_width, 3))
            * 255
        )
        top_down_map = np.concatenate((white, top_down_map), axis=1)
        pano_frame = np.concatenate((pano_frame, top_down_map), axis=0)

    return pano_frame.astype(np.uint8)


def add_id_on_img(img: ndarray, txt_id: str) -> ndarray:
    img_height = img.shape[0]
    img_width = img.shape[1]
    white = np.ones((10, img.shape[1], 3)) * 255
    img = np.concatenate((img, white), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    thickness = 2
    text_width = cv2.getTextSize(txt_id, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt_id,
        (start_width, img_height),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def colorize_draw_agent_and_fit_to_height(
    info: Dict[str, Any], 
    output_height: int,
    vis_info: Dict,
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """

    top_down_map = deepcopy(info["map"])

    if vis_info is not None:
        if 'nodes' in vis_info:
            for p in vis_info['nodes']:
                waypoint = [int(item) for item in p.split(',')]
                maps.draw_waypoint(top_down_map, waypoint, info["meters_per_px"], info["bounds"], maps.NODE)
        if 'ghosts' in vis_info:
            for p in vis_info['ghosts']:
                waypoint = [int(item) for item in p.split(',')]
                maps.draw_waypoint(top_down_map, waypoint, info["meters_per_px"], info["bounds"], maps.GHOST)
        if 'predict_ghost' in vis_info:
            maps.draw_waypoint(top_down_map, vis_info['predict_ghost'], info["meters_per_px"], info["bounds"], maps.PREDICT_GHOST)
        
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["fog_of_war_mask"]
    )
    map_agent_pos = info["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map

def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final

def planner_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    frame = cv2.copyMakeBorder(frame, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,0,0))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

def navigator_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    frame = append_text_to_image(frame, observations["instruction"]["text"])

    return frame


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    scene_id: str,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
):
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}{v:.2f}")

    video_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )
