from Agent import Agent, Memory
import tensorflow as tf
import numpy as np
import random
from collections import deque


class Deque(Memory):
    def __init__(self, pool=deque(), memory_size=50000):
        super(Deque, self).__init__(pool, memory_size)

    def store_transition(self, s, a, r, s_, terminal):
        self.pool.append((s, a, r, s_, terminal))
        if len(self.pool) > self.memory_size:
            self.pool.popleft()


class DQN_Agent(Agent):
    def __init__(self, action_cnt=2, learning_rate=1e-6, reward_decay=0.99, e_greedy=0.1, replace_target_iter=1000,
                 batch_size=32, observe_step=10000., explore_step=3000000., memory=Deque(), use_pre_weights=False,
                 save_path='./saved_dqn_model/'):

        super(DQN_Agent, self).__init__(action_cnt, learning_rate, reward_decay, e_greedy, replace_target_iter,
                                        batch_size, observe_step, explore_step, memory)
        # record average score per episode
        self.score_per_episode = 0
        self.score = tf.placeholder(tf.float32, [], name='score')
        self.summary_score = tf.summary.scalar('score_per_episode', self.score)
        self.loss_per_step = 0

        self.writer = tf.summary.FileWriter(save_path, self.sess.graph)
        self.merge_score = tf.summary.merge([self.summary_score])

        checkpoint = tf.train.get_checkpoint_state(save_path)
        if use_pre_weights and checkpoint:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

    def build_layers(self, var_scope, in_val, w_initializer, b_initializer):
        with tf.variable_scope(var_scope):
            with tf.variable_scope('conv1'):
                # 80,80,4 --> 20,20,32
                conv1 = tf.layers.conv2d(in_val, 32, 8, 4, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
                # 20,20,32 --> 10,10,32
                pool1 = tf.layers.max_pooling2d(conv1, 2, 2, 'same', name='pool1')
            with tf.variable_scope('conv2'):
                # 10,10,32 --> 5,5,64
                conv2 = tf.layers.conv2d(pool1, 64, 4, 2, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
            with tf.variable_scope('conv3'):
                # 5,5,64 --> 5,5,64
                conv3 = tf.layers.conv2d(conv2, 64, 3, 1, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
                # 5,5,64 --> 1600
                conv3_flatten = tf.reshape(conv3, [-1, 1600])
                # 1600 --> 512
            with tf.variable_scope('fcl1'):
                fcl1 = tf.layers.dense(conv3_flatten, 512, tf.nn.relu, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_fc1')
                # 512 --> 2 actions
                output = tf.layers.dense(fcl1, self.action_cnt, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer)
        return output

    def _build_model(self):
        self.s = tf.placeholder(tf.float32, [None, 80, 80, 4], 's')
        self.a = tf.placeholder(tf.float32, [None, self.action_cnt], 'a')
        self.s_ = tf.placeholder(tf.float32, [None, 80, 80, 4], 's_')

        w_initializer = tf.truncated_normal_initializer(0., 0.01)
        b_initializer = tf.constant_initializer(0.01)

        self.q_eval = self.build_layers('eval_net', self.s, w_initializer, b_initializer)
        self.q_next = self.build_layers('target_net', self.s_, w_initializer, b_initializer)

        with tf.variable_scope('y'):
            self.y = tf.placeholder(tf.float32, [None, ])
        with tf.variable_scope('q_eval_a'):
            self.q_eval_a = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.q_eval_a, name='TemporalDiff_error'))
        summary_loss = tf.summary.scalar('loss', self.loss)
        self.merge_loss = tf.summary.merge([summary_loss])
        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr, beta1=0.).minimize(self.loss)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        minibatch = random.sample(self.memory.pool, self.batch_size)
        s_t_batch = [row[0] for row in minibatch]
        a_t_batch = [row[1] for row in minibatch]
        r_t_batch = [row[2] for row in minibatch]
        s_t1_batch = [row[3] for row in minibatch]

        y_batch = []
        
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: s_t1_batch})
        
        for i in range(len(minibatch)):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(r_t_batch[i])
            else:
                y_batch.append(r_t_batch[i] + self.gamma * np.max(q_next[i]))

        _, loss, summary_loss = self.sess.run([self._train_op, self.loss, self.merge_loss],
                                            feed_dict={
                                                self.s: s_t_batch,
                                                self.a: a_t_batch,
                                                self.y: y_batch,
                                            })
        
        self.loss_per_step += loss
        if self.learn_step_counter % 100 == 0:
            self.loss_per_step = round(self.loss_per_step/100, 3)
            self.writer.add_summary(summary_loss, self.learn_step_counter)
            self.loss_per_step = 0

        self.learn_step_counter += 1
