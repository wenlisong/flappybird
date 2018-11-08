def resize_gray_binary(image, IMAGE_WIDTH, IMAGE_HEIGHT):
    import cv2
    # resize
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # bgr to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return image


def train(network):
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84
    finish_step = 10000000
    if network == 'dqn':
        IMAGE_WIDTH = 80
        IMAGE_HEIGHT = 80
        from dqn import DQN_Agent
        score_graph_path = './saved_dqn_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                        save_path=score_graph_path,
                        use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    elif network == 'doubledqn':
        IMAGE_WIDTH = 80
        IMAGE_HEIGHT = 80
        from double_dqn import  DoubleDQN_Agent
        score_graph_path = './saved_double_dqn_model/'
        rl = DoubleDQN_Agent(save_path=score_graph_path,
                            use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    elif network == 'mydqn':
        from mydqn import DQN_Agent
        score_graph_path = './saved_mydqn_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                        save_path=score_graph_path,
                        use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    elif network == 'mydqn2':
        from mydqn2 import DQN_Agent
        score_graph_path = './saved_mydqn2_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                        save_path=score_graph_path,)
        play2(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    elif network == 'priodqn':
        from prio_dqn import Prio_DQN_Agent
        score_graph_path = './saved_prio_dqn_model/'
        rl = Prio_DQN_Agent(learning_rate=1e-5,
                            save_path=score_graph_path,)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    elif network == 'pospriodqn':
        from pos_prio_dqn import Pos_Prio_DQN_Agent
        score_graph_path = './saved_pos_prio_dqn_model/'
        rl = Pos_Prio_DQN_Agent(learning_rate=1e-5,
                                save_path=score_graph_path,)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step)
    
    

def play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step):
    from game import wrapped_flappy_bird as fb
    import numpy as np
    env = fb.GameState()

    # first action [1,0], choose do nothing
    do_nothing = np.zeros(rl.action_cnt)
    do_nothing[0] = 1

    img, r_0, terminal = env.frame_step(do_nothing)

    # image preprocessing
    img = resize_gray_binary(img, IMAGE_WIDTH, IMAGE_HEIGHT)
    s_t = np.stack((img, img, img, img), axis=2)

    step = 0
    episode = 0
    while True:
        # rl choose action based on current state
        a_t = rl.choose_action(s_t)

        # rl take action and get next image and reward
        img, r_t, terminal = env.frame_step(a_t)
        
        if r_t == 1:
            rl.score_per_episode += 1
        if terminal:
            episode += 1
            if episode % 10 == 0:
                rl.score_per_episode = round(rl.score_per_episode/10, 3)
                summary, summary_score = rl.sess.run([rl.summary_score, rl.score], feed_dict={rl.score: rl.score_per_episode})
                rl.writer.add_summary(summary, episode)
                rl.score_per_episode =0

        img = resize_gray_binary(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        img = np.reshape(img, (IMAGE_WIDTH, IMAGE_HEIGHT, 1))

        # add new frame and delete the last one
        s_t1 = np.append(img, s_t[:, :, :3], axis=2)

        rl.memory.store_transition(s_t, a_t, r_t, s_t1, terminal)
        if step > rl.observe_step:
            rl.learn()
            rl.epsilon_annealing()

        # swap observation
        s_t = s_t1
        step += 1

        if step % 10000 == 0:
            rl.saver.save(rl.sess, score_graph_path + 'FLAPYBIRD-rl', global_step=step)
            # print('Save params at episode {0}'.format(step))

        state = ""
        if step <= rl.observe_step:
            state = "observe"
        elif rl.observe_step < step <= rl.observe_step + rl.explore_step:
            state = "explore"
        else:
            state = "train"
        print("STEP", step, "/ STATE", state, "/ EPSILON", rl.epsilon, "/ ACTION", np.argmax(a_t), "/ REWARD", r_t)

        if step > finish_step:
            break


def play2(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_step):
    from game import wrapped_flappy_bird as fb
    import numpy as np

    game_state = fb.GameState()

    # first action [1,0], choose do nothing
    do_nothing = np.zeros(rl.action_cnt)
    do_nothing[0] = 1

    img, r_0, terminal = game_state.frame_step(do_nothing)

    # image preprocessing
    img = resize_gray_binary(img, IMAGE_WIDTH, IMAGE_HEIGHT)
    s_t = np.stack((img, img, img, img), axis=2)
    x_pos = np.array([np.arange(0, IMAGE_WIDTH) for _ in range(IMAGE_HEIGHT)])/(IMAGE_WIDTH-1)
    y_pos = x_pos.transpose()
    pos_channel = np.stack((x_pos,y_pos), axis=2)

    step = 0
    episode = 0
    while True:
        # rl choose action based on current state
        a_t = rl.choose_action(np.concatenate((s_t, pos_channel), axis=2))

        # rl take action and get next image and reward
        img, r_t, terminal = game_state.frame_step(a_t)

        if r_t == 1:
            rl.score_per_episode += 1
        if terminal:
            episode += 1
            if episode % 10 == 0:
                rl.score_per_episode = round(rl.score_per_episode/10, 3)
                summary, summary_score = rl.sess.run([rl.summary_score, rl.score], feed_dict={rl.score: rl.score_per_episode})
                rl.writer.add_summary(summary, episode)
                rl.score_per_episode =0

        img = resize_gray_binary(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        img = np.reshape(img, (84, 84, 1))

        # add new frame and delete the last one
        s_t1 = np.append(img, s_t[:, :, :3], axis=2)

        rl.memory.store_transition(np.concatenate((s_t, pos_channel), axis=2), a_t, r_t, np.concatenate((s_t1, pos_channel), axis=2), terminal)
        if step > rl.observe_step:
            rl.learn()
            rl.epsilon_annealing()

        # swap observation
        s_t = s_t1
        step += 1

        if step % 10000 == 0:
            rl.saver.save(rl.sess, score_graph_path + 'FLAPYBIRD-rl', global_step=step)
            # print('Save params at episode {0}'.format(step))

        state = ""
        if step <= rl.observe_step:
            state = "observe"
        elif rl.observe_step < step <= rl.observe_step + rl.explore_step:
            state = "explore"
        else:
            state = "train"

        print("STEP", step, "/ STATE", state, "/ EPSILON", rl.epsilon, "/ ACTION", np.argmax(a_t), "/ REWARD", r_t)

        if step > finish_step:
            break


def main():
    train()

if __name__ == '__main__':
    main()