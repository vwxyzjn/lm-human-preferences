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

queries = np.array([
    [23073, 50259, 50259],
    [865, 234, 352],
    [20223, 8677, 50259],
])
responses = np.array([
    [837,   339,   561],
    [307, 24447,  1088],
    [262,  2877,  2119],
])
rewards = np.array([
    [1.2, 1.3, 1.4],
    [1.5, 1.6, 1.7],
    [1.8, 1.9, 2.0],
])
# responses = np.array([[837,   339,   561]])
comm = MPI.COMM_WORLD

with tf.Graph().as_default():
    hyperparams.dump(hparams)
    m = trained_models.TrainedModel(hparams.task.policy.initial_model)
    encoder = m.encoding.get_encoder()

    hparams.task.query_dataset="books"
    hparams.task.query_length = 3
    hparams.task.response_length = 3
    hparams.task.policy.temperature = 1.0

    policy = Policy(
        m, scope='policy',
        is_root=comm.Get_rank() == 0,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=1.0,
            build_respond=False)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.group(global_step.assign_add(1))

    @utils.graph_function()
    def sync_models():
        return utils.variable_synchronizer(comm, vars=policy.get_params())

    init_ops = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())
    

    def loss_fn(rollouts):
        values = rollouts['values']
        old_logprob = rollouts['logprobs']
        rewards = rollouts['rewards']
        with tf.name_scope('ppo_loss'):
            # if hparams.ppo.whiten_rewards:
            #     rewards = utils.whiten(rewards, shift_mean=False)
                # rewards = tf.Print(rewards, [rewards], 'whitened rewards', summarize=100)

            lastgaelam = 0
            advantages_reversed = []
            gen_length = hparams.task.response_length
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + hparams.ppo.gamma * nextvalues - values[:, t]
                # delta = tf.Print(delta, [t, delta, rewards[:, t], hparams.ppo.gamma * nextvalues, values[:, t]], 'delta')
                lastgaelam = delta + hparams.ppo.gamma * hparams.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = tf.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values

            # advantages = utils.whiten(advantages)
            advantages = tf.stop_gradient(advantages)  # Shouldn't do anything, but better not to think about it

            outputs = policy.analyze_responses_op(rollouts['queries'], rollouts['responses'])

            logprob = outputs['logprobs']
            ratio = tf.exp(logprob - old_logprob)
            ratio = tf.Print(ratio, [ratio], 'ratio', summarize=100)
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * tf.clip_by_value(ratio, 1.0 - hparams.ppo.cliprange, 1.0 + hparams.ppo.cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            pg_clipfrac = tf.reduce_mean(tf.cast(tf.greater(pg_losses2, pg_losses), tf.float32))

            loss = pg_loss # + hparams.ppo.vf_coef * vf_loss

            entropy = tf.reduce_mean(outputs['entropies'])
            approxkl = .5 * tf.reduce_mean(tf.square(logprob - old_logprob))

            return_mean, return_var = tf.nn.moments(returns, axes=list(range(returns.shape.ndims)))

            stats = dict(
                loss=dict(policy=pg_loss, total=loss),
                policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
                returns=dict(mean=return_mean, var=return_var),
            )
            return loss, utils.flatten_dict(stats, sep='/')

    


    with tf.Session() as sess:
        init_ops.run()
        sync_models()
        outputs = policy.analyze_responses(queries, responses)
        rollouts = {}
        rollouts['logprobs'] = outputs['logprobs']
        rollouts['queries'] = queries
        rollouts['responses'] = responses
        rollouts['values'] = tf.constant(outputs['values'], dtype=tf.float32)
        rollouts['rewards'] = tf.constant(rewards, dtype=tf.float32)
        print("outputs[logits]", outputs['logits'])
        print("outputs[logprobs]", outputs['logprobs'])


        # params = policy.get_params()
        # loss, stats = loss_fn(rollouts)
        # with tf.name_scope('ppo_opt', 'minimize'):
        #     with tf.name_scope('grads'):
        #         grads = tf.gradients(loss, params)
        #     grads, params = zip(*[(g, v) for g, v in zip(grads, params) if g is not None])
        #     optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, epsilon=1e-5, name='adam')
        #     opt_op = optimizer.apply_gradients(zip(grads, params), name='ppo_opt')
        # init_ops.run()
        for epoch in range(2):
            params = policy.get_params()
            loss, stats = loss_fn(rollouts)
            with tf.name_scope('ppo_opt', 'minimize'):
                with tf.name_scope('grads'):
                    grads = tf.gradients(loss, params)
                grads, params = zip(*[(g, v) for g, v in zip(grads, params) if g is not None])
                optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, epsilon=1e-5, name='adam')
                opt_op = optimizer.apply_gradients(zip(grads, params), name='ppo_opt')
                param_of_interest, grad_of_interest = sess.run((params[4], grads[4]))
                print("sess.run(grads)", params[4].name, param_of_interest, grad_of_interest)
                print(sess.run(stats))
                if epoch == 0:
                    sess.run(tf.global_variables_initializer())
                    g = sess.run(opt_op)
                # g = sess.run(opt_op)
        
        breakpoint()
        print("haha")
        # breakpoint()
        # tf.get_default_graph().finalize()
        