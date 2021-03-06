from dqn import DQN_Agent
import numpy as np
import random


class DoubleDQN_Agent(DQN_Agent):
    def __init__(self, e_greedy=0.1, use_pre_weights=False, save_path='./saved_double_dqn_model/'):
        super(DoubleDQN_Agent, self).__init__(e_greedy=e_greedy, use_pre_weights=use_pre_weights, save_path=save_path)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        minibatch = random.sample(self.memory.pool, self.batch_size)
        s_t_batch = [row[0] for row in minibatch]
        a_t_batch = [row[1] for row in minibatch]
        r_t_batch = [row[2] for row in minibatch]
        s_t1_batch = [row[3] for row in minibatch]

        y_batch = []

        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: s_t1_batch, self.s: s_t_batch})

        for i in range(len(minibatch)):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(r_t_batch[i])
            else:
                max_act = np.argmax(q_eval4next[i])
                y_batch.append(r_t_batch[i] + self.gamma * q_next[i, max_act])

        _, loss, summary_loss = self.sess.run([self._train_op, self.loss, self.merge_loss],
                                feed_dict={self.s: s_t_batch, self.a: a_t_batch, self.y: y_batch})

        self.loss_per_step += loss
        if self.learn_step_counter % 100 == 0:
            self.loss_per_step = round(self.loss_per_step/100, 3)
            self.writer.add_summary(summary_loss, self.learn_step_counter)
            self.loss_per_step = 0
        
        self.learn_step_counter += 1
