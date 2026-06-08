import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label

def norm_ang(angle):
    angle_t = (angle + np.pi) % (2 * np.pi) - np.pi
    return torch.tensor(angle_t)
        
def ang_diff(angle1, angle2):
    if not torch.is_tensor(angle1):
        angle1 = torch.tensor(angle1)
    if not torch.is_tensor(angle2):
        angle2 = torch.tensor(angle2)
    diff = torch.abs(angle1 - angle2)
    diff = diff % (2 * np.pi)
    diff = torch.min(diff, 2 * np.pi - diff)
    return torch.tensor([diff, angle1, angle2])

def extract_valid_mask_from_skeleton_bev(start_point, skeleton_bev):
    '''obtain the connected region where the target point is located 
    '''
    device = skeleton_bev.device
    np_arr = skeleton_bev.cpu().numpy().astype(np.int8)
    
    struct = np.ones((3,3), dtype=np.int8)
    labeled, _ = label(np_arr, structure=struct)

    _label = labeled[start_point[0], start_point[1]]
    mask = (labeled == _label)

    return torch.from_numpy(mask.copy()).to(device=device, dtype=torch.bool)

def edt(binary_map):
    device = binary_map.device
    np_map = binary_map.cpu().numpy()
    # consider the boundary
    padded_map = np.pad(np_map, pad_width=1, mode='constant', constant_values=0)
    dist_padded = distance_transform_edt(padded_map)
    dist = dist_padded[1:-1, 1:-1]

    return torch.from_numpy(dist.copy()).to(device)

class WeightedKMeans:
    """K-means clustering supporting weighted points
    """
    def __init__(self, n_clusters=8, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit_predict(self, X):
        interval = X.size(0) // self.n_clusters
        centroids = X[torch.arange(0, X.size(0), interval)]
        self.n_clusters = centroids.size(0)
        
        for _ in range(self.max_iter):
            # calculate the distances
            dists = torch.cdist(X, centroids)
            # assign cluster labels
            labels = dists.argmin(dim=1)

            # update the center
            new_centroids = []
            _centroids = []
            for i in range(self.n_clusters):
                x_label = X[labels == i]
                if len(x_label) > 0:
                    center = x_label.mean(dim=0)
                    new_centroids.append(center)
                    _centroids.append(centroids[i])

            new_centroids = torch.stack(new_centroids)
            _centroids = torch.stack(_centroids)

            # whether convergence
            if torch.allclose(_centroids, new_centroids, rtol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        dists = torch.cdist(X, centroids)
        labels = dists.argmin(dim=1)
        return labels