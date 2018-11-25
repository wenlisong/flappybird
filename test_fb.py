import numpy as np

def resize_gray_binary(image, IMAGE_WIDTH, IMAGE_HEIGHT):
    import cv2
    # resize
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # bgr to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return image

finish_episode = 1000

def run(network):
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84
    if network == 'dqn':
        IMAGE_WIDTH = 80
        IMAGE_HEIGHT = 80
        from dqn import DQN_Agent
        score_graph_path = './saved_dqn_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                       e_greedy=0.0,
                       save_path=score_graph_path,
                       use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'double_dqn':
        IMAGE_WIDTH = 80
        IMAGE_HEIGHT = 80
        from double_dqn import DoubleDQN_Agent
        score_graph_path = './saved_double_dqn_model/'
        rl = DoubleDQN_Agent(e_greedy=0.0,
                             save_path=score_graph_path,
                             use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'mydqn':
        from mydqn import DQN_Agent
        score_graph_path = './saved_mydqn_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                       e_greedy=0.0,
                       save_path=score_graph_path,
                       use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'mydqn2':
        from mydqn2 import DQN_Agent
        score_graph_path = './saved_mydqn2_model/'
        rl = DQN_Agent(learning_rate=1e-5,
                       e_greedy=0.0,
                       save_path=score_graph_path,
                       use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'prio_dqn':
        from prio_dqn import Prio_DQN_Agent
        score_graph_path = './saved_prio_dqn_model/'
        rl = Prio_DQN_Agent(learning_rate=1e-5,
                            e_greedy=0.0,
                            save_path=score_graph_path,
                            use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'pos_prio_dqn':
        from pos_prio_dqn import Pos_Prio_DQN_Agent
        score_graph_path = './saved_pos_prio_dqn_model/'
        rl = Pos_Prio_DQN_Agent(learning_rate=1e-5,
                                e_greedy=0.0,
                                save_path=score_graph_path,
                                use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)
    elif network == 'dueling_dqn':
        from dueling_dqn import Dueling_DQN_Agent
        score_graph_path = './saved_dueling_dqn_model/'
        rl = Dueling_DQN_Agent(learning_rate=1e-6,
                               e_greedy=0.0,
                               save_path=score_graph_path,
                               use_pre_weights=True)
        play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode)


def play1(rl, score_graph_path, IMAGE_WIDTH, IMAGE_HEIGHT, finish_episode):
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

    episode = 0
    score_hist = []
    while True:
        # rl choose action based on current state
        a_t = rl.choose_action(s_t)

        # rl take action and get next image and reward
        img, r_t, terminal = env.frame_step(a_t)

        if r_t == 1:
            rl.score_per_episode += 1
            print(rl.score_per_episode)
        if terminal:
            episode += 1
            rl.score_per_episode = round(rl.score_per_episode, 3)
            summary, summary_score = rl.sess.run([rl.summary_score, rl.score], feed_dict={
                                                    rl.score: rl.score_per_episode})
            rl.writer.add_summary(summary, episode)
            score_hist.append(rl.score_per_episode)
            rl.score_per_episode = 0
            if episode >= finish_episode:
                break

        img = resize_gray_binary(img, IMAGE_WIDTH, IMAGE_HEIGHT)
        img = np.reshape(img, (IMAGE_WIDTH, IMAGE_HEIGHT, 1))
        s_t1 = np.append(img, s_t[:, :, :3], axis=2)

        # swap observation
        s_t = s_t1
    max_score = max(score_hist)
    min_score = min(score_hist)
    aver_score = np.average(score_hist)
    std_deviation = np.std(score_hist)
    with open(score_graph_path + 'result.txt', 'w') as f:
        f.write('%s\n' % score_hist)
        f.write('max: %d\n' % max_score)
        f.write('min: %d\n' % min_score)
        f.write('average: %d\n' % aver_score)
        f.write('std deviation: %d\n' % std_deviation)

def main():
    test()


if __name__ == '__main__':
    main()
