import sys
import getopt
import pdb


def resize_gray_binary(image):
    import cv2
    # resize
    image = cv2.resize(image, (80, 80))
    # bgr to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return image


def train(rl, score_graph_path):
    import os
    current_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(current_dir)
    from game import wrapped_flappy_bird as fb
    import numpy as np

    game_state = fb.GameState()

    # first action [1,0], choose do nothing
    do_nothing = np.zeros(rl.n_actions)
    do_nothing[0] = 1

    img, r_0, terminal = game_state.frame_step(do_nothing)
    rl.score_per_episode += r_0
    # image preprocessing
    img = resize_gray_binary(img)
    s_t = np.stack((img, img, img, img), axis=2)

    step = 0
    episode = 0
    while True:
        # rl choose action based on current state
        a_t = rl.choose_action(s_t)

        # rl take action and get next image and reward
        img, r_t, terminal = game_state.frame_step(a_t)

        if not terminal:
            rl.score_per_episode += r_t
        else:
            # print(episode, rl.score_per_episode)
            if episode % 10 == 0:
                summary, score = rl.sess.run([rl.merge_op, rl.score], feed_dict={rl.score: rl.score_per_episode})
                rl.writer.add_summary(summary, episode)
            rl.score_per_episode = 0
            episode += 1

        img = resize_gray_binary(img)
        img = np.reshape(img, (80, 80, 1))

        # add new frame and delete the last one
        s_t1 = np.append(img, s_t[:, :, :3], axis=2)

        rl.store_transition(s_t, a_t, r_t, s_t1, terminal)

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


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hn:", ["help", "network="])
        except getopt.error as msg:
            raise Usage(msg)
    except Usage as err:
        print(sys.stderr, err.msg)
        print(sys.stderr, "for help use --help")
        return 2

    if len(opts) == 0:
        print("Please specify parameters!")
        return 1
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(__doc__)
            return 0
        elif opt in ("-n", "--network"):
            if arg == 'dqn':
                from dqn import DeepQNetwork
                score_graph_path = './saved_model/'
                network = DeepQNetwork(e_greedy=0.1,
                                       output_graph=True,
                                       save_path=score_graph_path,)
            elif arg == 'doubledqn':
                from double_dqn import DoubleDQN
                score_graph_path = './saved_model_doubledqn/'
                network = DoubleDQN(e_greedy=0.1,
                                    output_graph=True,
                                    save_path=score_graph_path,)

            else:
                print("You could choose 'dqn', 'doubledqn' as network's parameter")
                return 1
            train(network, score_graph_path)
            return 0


if __name__ == "__main__":
    sys.exit(main())
