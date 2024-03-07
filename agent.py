import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class ActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        self.log_std_dist = Uniform(torch.tensor([-0.01]), torch.tensor([0.01]))
        super(ActorCriticPolicy, self).__init__(*args, **kwargs,
                                                net_arch={'vf': [1024, 512], 'pi': [1024, 512]},
                                                activation_fn=nn.ReLU,
                                                log_std_init=self.log_std_dist.sample().float())

        # Manually update the optimizer to enable per-layer hyperparameters to be set.
        self.optimizer = torch.optim.SGD([{'params': self.pi_features_extractor.parameters(),
                                           'lr': 0.000002,
                                           'weight_decay': 0.0005},
                                          {'params': self.vf_features_extractor.parameters(),
                                           'lr': 0.0005,
                                           'weight_decay': 0.}], momentum=0.9)
