#!/usr/bin/env python3

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.contrib import summary

from lm_human_preferences import lm_tasks, train_reward
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.rewards import TrainedRewardModel
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams
from lm_human_preferences.utils.core import Schema


@dataclass
class AdaptiveKLParams(hyperparams.HParams):
    target: float = None
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams(hyperparams.HParams):
    kl_coef: float = 0.2
    adaptive_kl: Optional[AdaptiveKLParams] = None

    trained_model: Optional[str] = None

    train_new_model: Optional[train_reward.HParams] = None

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
        assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'


@dataclass
class PpoHParams(hyperparams.HParams):
    total_episodes: int = 2000000
    batch_size: int = 64
    nminibatches: int = 1
    noptepochs: int = 4
    lr: float = 5e-6
    vf_coef: float = .1
    cliprange: float = .2
    cliprange_value: float = .2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


@dataclass
class HParams(hyperparams.HParams):
    run: train_reward.RunHParams = field(default_factory=train_reward.RunHParams)

    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)
    task_id: str = None

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        # NOTE: must additionally divide by # ranks
        minibatch_size = utils.exact_div(self.ppo.batch_size, self.ppo.nminibatches)
        if self.ppo.whiten_rewards:
            assert minibatch_size >= 8, \
                f"Minibatch size {minibatch_size} is insufficient for whitening in PPOTrainer.loss"



hparams = HParams()
save_dir = hparams.run.save_dir
if hparams.rewards.train_new_model:
    assert hparams.task == hparams.rewards.train_new_model.task, f'{hparams.task} != {hparams.rewards.train_new_model.task}'
    hparams.rewards.train_new_model.run.save_dir = save_dir
    train_reward.train(hparams.rewards.train_new_model)
    if 'pytest' in sys.modules:
        hparams.rewards.trained_model = 'test'
    elif save_dir:
        hparams.rewards.trained_model = None if save_dir is None else os.path.join(save_dir, 'reward_model')

# oai_example_query_response = [23073,   837,   339,   561,   307, 24447,  1088,   262,  2877,  2119, 837,  2712,   351,   465, 14958,   764, 50259, 50259, 50259, 50259,
# 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
# 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
# 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
# 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
# 50259, 50259, 50259, 50259,  1212,   318,   845,  1107, 15774,   355,
# 314,   588,   616, 12575,  1525,   284,  1394,   465,  1986,   284,
# 2241,   290,   465, 14958,   284,  2241,   764,   314]
# oai_example_query_response = np.array([oai_example_query_response])
# context_length = 64
# responses = oai_example_query_response[:,context_length:]
# queries = oai_example_query_response[:,:context_length]

queries = np.array([[23073]])
responses = np.array([[837,   339,   561]])
comm = MPI.COMM_WORLD
with tf.Graph().as_default():
    hyperparams.dump(hparams)
    m = trained_models.TrainedModel(hparams.task.policy.initial_model)
    encoder = m.encoding.get_encoder()

    hparams.task.query_dataset="books"
    hparams.task.query_length = 64
    hparams.task.response_length = 24
    hparams.task.policy.temperature = 1.0

    ref_policy = Policy(
        m, scope='ref_policy',
        is_root=comm.Get_rank() == 0,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=1.0,
            build_respond=False)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.group(global_step.assign_add(1))

    init_ops = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())
    
    with tf.Session() as sess:
        init_ops.run()

        for i in range(1):
            output = ref_policy.analyze_responses(queries, responses)
            print("logprobs", output['logprobs'])
            print("all_logits", output['all_logits'], output['all_logits'].shape)
            entropy = tf.reduce_sum(-output['logprobs'], axis=1)
            print("sess.run(entropy)", sess.run(entropy))

        tf.get_default_graph().finalize()
        