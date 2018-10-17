import tensorflow as tf
import numpy as np
from collections import deque
import random
import pdb

np.random.seed(2)
tf.set_random_seed(2)


class DeepQNetwork:
    def __init__(self, n_actions=2, learning_rate=0.5, reward_decay=0.99, e_greedy=0.9, replace_target_iter=4,
                 memory_size=500, batch_size=32, observe_step=100., explore_step=5000.,
                 output_graph=False, use_pre_weights=False):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.observe_step = observe_step
        self.explore_step = explore_step
        self.init_epsilon = e_greedy
        self.final_epsilon = 0.0001

        self.learn_step_counter = 0
        self.memory = deque()

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
        self.saver = tf.train.Saver(max_to_keep=2)

        # record average score per episode
        self.score_per_episode = 0
        self.score = tf.placeholder(tf.float16, [])
        if output_graph:
            self.writer = tf.summary.FileWriter('./net_graph', self.sess.graph)
            self.summary = tf.summary.scalar('score_per_episode', self.score)
            self.merge_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("./saved_net_weight")
        if use_pre_weights and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def epsilon_annealing(self, step):
        if self.epsilon > self.final_epsilon and step > self.observe_step:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore_step

    def _build_net(self):
        # input
        self.s = tf.placeholder(tf.float32, [None, 80, 80, 4], 's')
        self.a = tf.placeholder(tf.float32, [None, 2], 'a')
        self.r = tf.placeholder(tf.float32, [None, ], 'r')
        self.s_ = tf.placeholder(tf.float32, [None, 80, 80, 4], 's_')

        w_initializer = tf.truncated_normal_initializer(0., 0.01)
        b_initializer = tf.constant_initializer(0.01)

        # eval_net
        with tf.variable_scope('eval_net'):
            # 84,84,4 --> 20,20,32
            h_conv1 = tf.layers.conv2d(self.s, 32, 8, 4, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv1')
            # 20,20,32 --> 10,10,32
            h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2, 'same', name='h_pool1')
            # 10,10,32 --> 5,5,64
            h_conv2 = tf.layers.conv2d(h_pool1, 64, 4, 2, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv2')
            # 5,5,64 --> 5,5,64
            h_conv3 = tf.layers.conv2d(h_conv2, 64, 3, 1, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv3')
            # 5,5,64 --> 1600
            h_conv3_flatten = tf.reshape(h_conv3, [-1, 1600])
            # 1600 --> 512
            h_fc1 = tf.layers.dense(h_conv3_flatten, 512, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='h_fc1')
            # 512 --> 2 actions
            self.q_eval = tf.layers.dense(h_fc1, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_evel')

        # target net
        with tf.variable_scope('target_net'):
            # 84,84,4 --> 20,20,32
            h_conv1 = tf.layers.conv2d(self.s, 32, 8, 4, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv1')
            # 20,20,32 --> 10,10,32
            h_pool1 = tf.layers.max_pooling2d(h_conv1, 2, 2, 'same', name='h_pool1')
            # 10,10,32 --> 5,5,64
            h_conv2 = tf.layers.conv2d(h_pool1, 64, 4, 2, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv2')
            # 5,5,64 --> 5,5,64
            h_conv3 = tf.layers.conv2d(h_conv2, 64, 3, 1, 'same', activation=tf.nn.relu,
                                       kernel_initializer=w_initializer, bias_initializer=b_initializer, name='h_conv3')
            # 5,5,64 --> 1600
            h_conv3_flatten = tf.reshape(h_conv3, [-1, 1600])
            # 1600 --> 512
            h_fc1 = tf.layers.dense(h_conv3_flatten, 512, tf.nn.relu, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='h_fc1')
            # 512 --> 2 actions
            self.q_next = tf.layers.dense(h_fc1, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='q_max_s_')
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            self.q_eval_a = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)
        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_a, name='TemporalDiff_error'))

        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def store_transition(self, s, a, r, s_, terminal):
        self.memory.append((s, a, r, s_, terminal))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = np.zeros([self.n_actions])
        if np.random.uniform() > self.epsilon:
            action_val = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_idx = np.argmax(action_val)
        else:
            action_idx = np.random.randint(0, self.n_actions)
        action[action_idx] = 1

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('target params are replaced at step {0}'.format(self.learn_step_counter))

        minibatch = random.sample(self.memory, self.batch_size)

        s_t_batch = [row[0] for row in minibatch]
        a_t_batch = [row[1] for row in minibatch]
        r_t_batch = [row[2] for row in minibatch]
        s_t1_batch = [row[3] for row in minibatch]

        _, cost = self.sess.run([self._train_op, self.cost],
                                feed_dict={
                                    self.s: s_t_batch,
                                    self.a: a_t_batch,
                                    self.r: r_t_batch,
                                    self.s_: s_t1_batch,
                                })

        self.learn_step_counter += 1


if __name__ == '__main__':
    DeepQNetwork(output_graph=True)
