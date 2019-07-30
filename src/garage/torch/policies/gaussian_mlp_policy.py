"""GaussianMLPPolicy."""

import torch

from garage.torch.policies.base import Policy


class GaussianMLPPolicy(Policy):
    """
    GaussianMLPPolicy.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        module : GaussianMLPModule to make prediction based on a gaussian
        distribution.
    :return:
    """

    def __init__(self, env_spec, module):
        self._module = module

    def forward(self, inputs):
        """Forward method."""
        return self._module(torch.Tensor(inputs))

    def get_actions(self, observations):
        """Get actions given observations."""
        with torch.no_grad():
            dist = self.forward(observations)
            return dist.rsample().numpy(), {}

    def get_log_likelihood(self, observation, action):
        """Get log likelihood given observations and action."""
        dist = self.forward(observation)
        return dist.log_prob(action)

    def get_entropy(self, observation):
        """Get entropy given observations."""
        dist = self.forward(observation)
        return dist.entropy()

    def get_module(self):
        """Get gaussinan mlp module."""
        return self._module

    def reset(self, dones=None):
        """Reset the environment."""
        pass

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True
