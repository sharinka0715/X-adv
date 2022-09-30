import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class Actor(nn.Module):
    def __init__(self, patch_count=4, n_actions=50):
        super(Actor, self).__init__()
        self.patch_count = patch_count
        self.n_actions = n_actions
        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.act = nn.Linear(256, patch_count * n_actions)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.act(x)
        return out.view(-1, self.patch_count, self.n_actions)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), device="cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count