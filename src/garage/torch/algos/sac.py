"""This modules creates a sac model in PyTorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np


class SAC(OffPolicyRLAlgorithm):
    """ A SAC Model in Torch.

    Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 target_entropy=None,
                 use_automatic_entropy_tuning=True,
                 discount=0.99,
                 n_epoch_cycles=20,
                 n_train_steps=50,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 exploration_strategy=None,
                 target_update_tau=1e-2,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 optimizer=torch.optim.Adam,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 max_action=None,
                 reward_scale=1.,
                 smooth_return=True,
                 input_include_goal=False):

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2

        action_bound = env_spec.action_space.high
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.max_action = action_bound if max_action is None else max_action
        self.evaluate = False
        self.input_include_goal = input_include_goal

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=n_train_steps,
                         n_epoch_cycles=n_epoch_cycles,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         rollout_batch_size=rollout_batch_size,
                         exploration_strategy=exploration_strategy,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         reward_scale=reward_scale,
                         smooth_return=smooth_return)

        self.target_policy = copy.deepcopy(self.policy)
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)

        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()

    # 0) update policy using updated min q function
    # 1) compute targets from Q functions
    # 2) update Q functions using optimizer
    # 3) query Q functons, take min of those functions
    def train_once(self, itr, paths):
        """
        """
        # add paths to replay buffer

        # add new transitions to the replay buffer
        
    def optimize_policy(self, itr, samples):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.
        """
        pass

    def update_target(self):
        """Update parameters in the target policy and Q-value network."""
        for t_param, param in zip(self.target_qf.parameters(),
                                  self.qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

        for t_param, param in zip(self.target_policy.parameters(),
                                  self.policy.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

    def _add_transitions(self, paths):
        """ Add dictionary of paths(experience) to replay buffer.

            Args:
                - paths (dict): dictionary containing transition info
                                e.g. s,a,r,s', terminal state, goal state
                                
        """
        paths = self.process_samples(itr, paths)
        epoch = itr / self.n_epoch_cycles

        rewards = paths['rewards']
        terminals = paths['terminals']
        obs = paths['observations']
        actions = paths['actions']
        next_obs = paths['next_obs']
        # check if these exist in the patgh
        goal = paths['goal']
        achieved_goal = paths['achieved_goal']

        if self.input_include_goal:
            self.replay_buffer.add_transitions(
                observation=obs,
                action=actions,
                goal=d_g,
                achieved_goal=a_g,
                terminal=dones,
                next_observation=[
                    next_obs['observation'] for next_obs in next_obses
                ],
                next_achieved_goal=[
                    next_obs['achieved_goal'] for next_obs in next_obses
                ],
            )
        else:
            self.algo.replay_buffer.add_transitions(
                observation=obses,
                action=actions,
                reward=rewards * self.algo.reward_scale,
                terminal=dones,
                next_observation=next_obses,
            )
