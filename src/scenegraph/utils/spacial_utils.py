import numpy as np

def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    # concern that panorama will be the set of 12 pictures of 90 HFOV
    yc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))

    camera_matrix = []
    for i in range(0,12,1):
        xc = 0.134 * width * (2 * i + 1)
        camera_matrix.append(
            np.array([
            [f, 0, xc],
            [0, f, yc],
            [0, 0, 1]
        ]))
    
    return camera_matrix

def get_pose_matrix(observation):
    '''camera coordinate to world coordinate, in right-handed system
    '''
    pose_matrixs = []
    x = observation['gps'][0]
    y = -observation['gps'][1]
    # camera to the agent
    camera2agent = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ])
    # agent to world
    extra_rads = [i/180*np.pi for i in range(0,360,30)]
    for extra_rad in extra_rads:
        t = (observation['compass'])[0] - extra_rad
        agent2world = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t), np.cos(t), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        pose_matrix = agent2world @ camera2agent
        pose_matrixs.append(pose_matrix)
    
    pose_matrixs = np.stack(pose_matrixs, axis=0)
    return pose_matrixs

def get_maskview_num(mask, width_single_view):
    width_index = np.where(mask>0)[1]
    if len(width_index) > 0:
        width_mean = np.mean(width_index)
        return int(width_mean // width_single_view)
    else:
        return -1


