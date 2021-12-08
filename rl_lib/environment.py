import gym
import Levenshtein
import gym
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import time
from itertools import chain
from torch.distributions import MultivariateNormal, Categorical
from tqdm import tqdm
import os
from collections import namedtuple
from torch import manual_seed
import torch.optim as optim
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.rna_lib import random_init_sequence_pair, structure_dotB2Edge, forbidden_actions_hike, rna_act_pair_hike, renew_forbidden_actions
import pathos.multiprocessing as pathos_mp
from functools import partial
import RNA

def multiply(x, y):
    return x * y

class RNA_Env(gym.Env):
    def __init__(self, dotB, action_space=4, pool=None, done_list=[]):
        super(RNA_Env, self).__init__()
        self.aim_dotB = dotB
        self.aim_edge = structure_dotB2Edge(dotB)
        self.l = len(self.aim_dotB)
        self.action_space = action_space
        # self.seq_base = random_init_sequence_pair(self.aim_dotB, self.aim_edge, self.l, self.action_space)[0]
        # self.last_distance = self.get_distance(self.seq_base)
        # self.forbidden_actions = forbidden_actions_hike(self.seq_base, self.aim_dotB, self.aim_edge, self.action_space)
        self.seq_base = ''
        self.last_distance = self.l
        self.last_novelty = 0.
        self.forbidden_actions = []
        if pool is None:
            self.pool = pathos_mp.ProcessingPool()
        else:
            self.pool = pool

        self.done_list = done_list

    def reset(self):
        self.seq_base = random_init_sequence_pair(self.aim_dotB, self.aim_edge, self.l, self.action_space)[0]
        self.last_distance = self.get_distance(self.seq_base)
        self.last_novelty = self.check_novelty(self.seq_base)
        self.forbidden_actions = forbidden_actions_hike(self.seq_base, self.aim_dotB, self.aim_edge, self.action_space)

    def get_distance(self, seq_base):
        real_dotB = RNA.fold(seq_base)[0]
        distance = Levenshtein.distance(real_dotB, self.aim_dotB)
        return distance

    def get_near_seq(self):
        actions = [i for i in range(self.l * self.action_space) if i not in self.forbidden_actions]
        near_work = partial(rna_act_pair_hike, seq_=self.seq_base, edge_index=self.aim_edge, action_space=self.action_space)
        near_seq_list = self.pool.map(near_work, actions)

        return near_seq_list, actions

    def get_near_distance(self, near_seq_list):
        near_distance_list = self.pool.map(self.get_distance, near_seq_list)
        return near_distance_list

    def get_energy(self, seq):
        energy = RNA.energy_of_struct(seq, self.aim_dotB)
        return energy

    def get_near_energy(self, near_seq_list):
        near_energy_list = self.pool.map(self.get_energy, near_seq_list)
        return near_energy_list

    def get_near_novelty(self, near_seq_list):
        near_novelty_list = self.pool.map(self.check_novelty, near_seq_list)
        return near_novelty_list

    def get_next_seq_order(self, near_distance_list, near_novelty_list, type='max'):
        near_dist = np.array(near_distance_list)
        near_dist_delta = self.last_distance - near_dist
        near_novel = np.array(near_novelty_list)
        near_score_list = near_dist_delta * near_novel
        near_prob = F.softmax(torch.tensor(near_score_list).float())
        if type == 'max':
            near_order = torch.argmax(near_prob)
        else:
            near_order = torch.multinomial(near_prob, 1)

        return near_order

    def renew_forbidden(self, action):
        self.forbidden_actions = renew_forbidden_actions(action, self.seq_base, self.aim_edge, self.forbidden_actions, self.action_space)

    def renew_env(self, next_seq, action, next_distance, next_novelty):
        self.forbidden_actions = renew_forbidden_actions(action, self.seq_base, self.aim_edge, self.forbidden_actions,
                                                         self.action_space)
        self.seq_base = next_seq
        self.last_distance = next_distance
        self.last_novelty = next_novelty

    def check_novelty(self, seq):
        if len(self.done_list) < 1:
            novelty = 1.
        else:
            text_dis_list = [Levenshtein.distance(seq, done_seq) for done_seq in self.done_list]
            dist_min = min(text_dis_list)
            novelty = dist_min / self.l
        return novelty





