from Agent import Agent, Memory
import tensorflow as tf
import numpy as np
import random


IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84

class Sum_Tree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
    
class Prio_Memory(Memory):
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, pool=None, memory_size=500000):
        self.pool = Sum_Tree(memory_size)
        super(Prio_Memory, self).__init__(self.pool, memory_size)

    def store_transition(self, s, a, r, s_, terminal):
        transition = (s, a, r, s_, terminal)
        max_p = np.max(self.pool.tree[-self.pool.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.pool.add(max_p, transition)   # set the max p for new p

    def sample(self, batch_size):
        # (32,) (32,5) (32,1)
        b_idx, b_memory, ISWeights = np.empty((batch_size,), dtype=np.int32), np.empty((batch_size, len(self.pool.data[0])), dtype=object), np.empty((batch_size, 1))
        pri_seg = self.pool.total_p / batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.pool.tree[-self.pool.capacity:]) / self.pool.total_p     # for later calculate ISweight
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.pool.get_leaf(v)
            prob = p / self.pool.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i,:] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.pool.update(ti, p)


class Pos_Prio_DQN_Agent(Agent):
    def __init__(self, action_cnt=2, learning_rate=1e-6, reward_decay=0.99, e_greedy=0.1, replace_target_iter=1000,
                 batch_size=32, observe_step=100., explore_step=3000000., memory=Prio_Memory(), use_pre_weights=True,
                 save_path='./saved_prio_dqn_model/'):

        super(Pos_Prio_DQN_Agent, self).__init__(action_cnt, learning_rate, reward_decay, e_greedy, replace_target_iter,
                                        batch_size, observe_step, explore_step, memory)
        # record average score per episode
        self.score_per_episode = 0
        self.score = tf.placeholder(tf.float16, [], name='score')
        self.summary_score = tf.summary.scalar('score_per_episode', self.score)
        self.loss_per_step = 0

        self.writer = tf.summary.FileWriter(save_path, self.sess.graph)
        self.merge_score = tf.summary.merge([self.summary_score])

        checkpoint = tf.train.get_checkpoint_state(save_path)
        if use_pre_weights and checkpoint:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

    def choose_action(self, observation):
        x_pos = np.array([np.arange(0, IMAGE_WIDTH) for _ in range(IMAGE_HEIGHT)])/(IMAGE_WIDTH-1)
        y_pos = x_pos.transpose()
        pos_channel = np.stack((x_pos, y_pos), axis=2)
        observation = np.concatenate((observation, pos_channel), axis=2)[np.newaxis, :]
        action = np.zeros([self.action_cnt], dtype=float)
        if np.random.uniform() > self.epsilon:
            action_val = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_idx = np.argmax(action_val, axis=1)
            # print('Q_Max_val {0}'.format(action_val))
        else:
            action_idx = np.random.randint(0, self.action_cnt)
        action[action_idx] = 1.

        return action

    def build_layers(self, var_scope, in_val, w_initializer, b_initializer):
        with tf.variable_scope(var_scope):
            with tf.variable_scope('conv1'):
                # 84,84,6 --> 42,42,64
                conv1 = tf.layers.conv2d(in_val, 64, 7, 2, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
                # 42,42,64 --> 20,20,64
                pool1 = tf.layers.max_pooling2d(conv1, 3, 2)
            with tf.variable_scope('conv2'):
                # 20,20,64 --> 20,20,128
                conv2 = tf.layers.conv2d(pool1, 128, 3, 1, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
                # 20,20,128 --> 9,9,128
                pool2 = tf.layers.max_pooling2d(conv2, 3, 2)
            with tf.variable_scope('conv3'):
                # 9,9,128 --> 9,9,256
                conv3 = tf.layers.conv2d(pool2, 256, 3, 1, 'same', activation=tf.nn.relu,
                                         kernel_initializer=w_initializer, bias_initializer=b_initializer)
                # 9,9,256 --> 4,4,256
                pool3 = tf.layers.average_pooling2d(conv3, 3, 2)
                # 4,4,256 --> 4096
                conv3_flatten = tf.reshape(pool3, [-1, 4096])
                # 4096 --> 1024
            with tf.variable_scope('fcl1'):
                fcl1 = tf.layers.dense(conv3_flatten, 1024, tf.nn.relu, kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer, name='e_fc1')
                # 1024 --> 2 actions
                output = tf.layers.dense(fcl1, self.action_cnt, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer)
        return output

    def _build_model(self):
        self.s = tf.placeholder(tf.float32, [None, 84, 84, 6], 's')
        self.a = tf.placeholder(tf.float32, [None, self.action_cnt], 'a')
        self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 6], 's_')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        w_initializer = tf.truncated_normal_initializer(0., 0.01)
        b_initializer = tf.constant_initializer(0.01)

        self.q_eval = self.build_layers('eval_net', self.s, w_initializer, b_initializer)
        self.q_next = self.build_layers('target_net', self.s_, w_initializer, b_initializer)

        with tf.variable_scope('y'):
            self.y = tf.placeholder(tf.float32, [None, ])
        with tf.variable_scope('q_eval_a'):
            self.q_eval_a = tf.reduce_sum(tf.multiply(self.q_eval, self.a), reduction_indices=1)
        with tf.name_scope('loss'):
            self.abs_errors = tf.abs(self.y - self.q_eval_a)
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.y, self.q_eval_a, name='TemporalDiff_error'))
        summary_loss = tf.summary.scalar('loss', self.loss)
        self.merge_loss = tf.summary.merge([summary_loss])
        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
        x_pos = np.array([np.arange(0, IMAGE_WIDTH) for _ in range(IMAGE_HEIGHT)])/(IMAGE_WIDTH-1)
        y_pos = x_pos.transpose()
        pos_channel = np.stack((x_pos, y_pos), axis=2)
        s_t_batch = [np.concatenate((row[0], pos_channel), axis=2) for row in minibatch]
        a_t_batch = [row[1] for row in minibatch]
        r_t_batch = [row[2] for row in minibatch]
        s_t1_batch = [np.concatenate((row[3], pos_channel), axis=2) for row in minibatch]

        y_batch = []
        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: s_t1_batch, self.s: s_t_batch})

        for i in range(len(minibatch)):
            terminal = minibatch[i, 4]
            if terminal:
                y_batch.append(r_t_batch[i])
            else:
                max_act = np.argmax(q_eval4next[i])
                y_batch.append(r_t_batch[i] + self.gamma * q_next[i, max_act])

        _, loss, summary_loss, abs_errors = self.sess.run([self._train_op, self.loss, self.merge_loss, self.abs_errors],
                                                            feed_dict={
                                                                self.s: s_t_batch,
                                                                self.a: a_t_batch,
                                                                self.y: y_batch,
                                                                self.ISWeights: ISWeights
                                                            })
        
        self.memory.batch_update(tree_idx, abs_errors)
        
        self.loss_per_step += loss
        if self.learn_step_counter % 100 == 0:
            self.loss_per_step = round(self.loss_per_step/100, 3)
            self.writer.add_summary(summary_loss, self.learn_step_counter)
            self.loss_per_step = 0
        
        self.learn_step_counter += 1
 