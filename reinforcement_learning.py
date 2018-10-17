import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self, pool=None, memory_size=1000):
        self.memory_size = memory_size
        self.pool = pool

    def store_transition(self, *args):
        pass


class ReinforcementLearning:
    def __init__(self, action_cnt=4, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, replace_target_iter=300,
                 batch_size=32, observe_step=200., explore_step=1000., memory=None):
        self.action_cnt = action_cnt
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.init_epsilon = e_greedy
        self.final_epsilon = 0.0001
        self.batch_size = batch_size
        self.observe_step = observe_step
        self.explore_step = explore_step
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        self.memory = memory

        self._build_model()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        # config gpu
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.allow_soft_placement=True
        # config.gpu_options.visible_device_list = '0'
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(max_to_keep=2)
        self.sess.run(tf.global_variables_initializer())

    def epsilon_annealing(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.explore_step

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        action = np.zeros([self.action_cnt], dtype=float)
        if np.random.uniform() > self.epsilon:
            action_val = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_idx = np.argmax(action_val, axis=1)
            print('Q_Max_val {0}'.format(action_val))
        else:
            action_idx = np.random.randint(0, self.action_cnt)
        action[action_idx] = 1.

        return action

    def _build_model(self):
        self.build_layers()

    def build_layers(self, *args):
        pass

    def learn(self, *args):
        pass
