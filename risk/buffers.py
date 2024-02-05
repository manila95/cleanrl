import torch
import numpy as np


class RiskBuffer:
    def __init__(self, fear_radius=99999, max_dist=1000):
        super().__init__()
        self.keys = ["obs", "actions", "next_obs", "dist_to_fail"]
        self.data = {}
        for key in self.keys:
            self.data[key] = None

        self.buffer_size = 0
        self.fear_radius = fear_radius
        self.max_dist = max_dist

    def __len__(self):
        return self.buffer_size

    def add(self, traj, terminated, traj_len):
        dist_to_fail = np.array(reversed(list(range(traj_len))) if terminated else  [self.max_dist]*traj_len)
        traj["dist_to_fail"] = dist_to_fail 

        idx = (dist_to_fail <= self.fear_radius)
        for key in self.keys:
            traj[key] = traj[key][idx] if traj[key] is None else np.concatenate([self.data[key], traj[key][idx]], 1) 
        self.buffer_size += np.sum(idx)

    def sample(self, sample_size):
        sample_idx = np.random.choice(range(self.buffer_size), sample_size, replace=False)
        return {key: self.data[key][sample_idx] for key in self.keys}



