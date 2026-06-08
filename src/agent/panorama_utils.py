import cv2
import numpy as np
import sys

def rgbs_to_panorama(observation):
    '''stitch the 12 rgb images which named as rgb or rgb_{degree}, and the HFOV is 90
    :param observation: the habitat-sim observation
    :output: panoramas[the panorama of rgb, the panorama of depth]
    '''
    panoramas = []
    if 'rgb' in observation:
        width = observation['rgb'].shape[1]
        width_1d3 = int(width * 0.366)
        width_2d3 = int(width * 0.634)
        images = [observation['rgb']]
        for i in range(30, 360, 30):
            images.append(observation[f"rgb_{i}"])
        # if the cv2 stitcher fail, just stitch simply
        for i in range(len(images)):
            images[i] = images[i][:,width_1d3:width_2d3,:]
        panorama = np.concatenate(images, axis=1)
        # add the first image at the end of the panoramic view
        panoramas.append(panorama)

    if 'depth' in observation:
        width = observation['depth'].shape[1]
        width_1d3 = int(width * 0.366)
        width_2d3 = int(width * 0.634)
        images = [observation['depth']]
        for i in range(30, 360 ,30):
            images.append(observation[f"depth_{i}"])
        
        # if the cv2 stitcher fail, just stitch simply
        for i in range(len(images)):
            images[i] = images[i][:,width_1d3:width_2d3,:]
        panorama = np.concatenate(images, axis=1)
        # add the first image at the end of the panoramic view
        panoramas.append(panorama)

    return panoramas

def rotate_180(panoramas, camera_matrix, pose_matrix):
    for i in range(len(panoramas)):
        h, w = panoramas[i].shape[0], panoramas[i].shape[1]
        split = int(w/2)
        panoramas[i] = np.concatenate([panoramas[i][:,split:,:], panoramas[i][:,:split,:]], axis=1)

    split = int(len(camera_matrix)/2)
    pose_matrix = np.concatenate([pose_matrix[split:,:], pose_matrix[:split,:]], axis=0)

    return panoramas, camera_matrix, pose_matrix