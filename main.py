import sys
import getopt


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hg:n:", ["help", "game=", "network="])
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
        elif opt in ("-g", "--game"):
            if arg == 'fb':
                from train_fb import train
            elif arg == 'fb2':
                from train_fb2 import train
            else:
                print("No this game, now we have flappy bird(fb)")
                return 1
        elif opt in ("-n", "--network"):
            train(arg)
            # if arg == 'dqn':
            #     train(arg)
            # elif arg == 'doubledqn':
            #     train(arg)
            # elif arg == 'mydqn':
            #     train(arg)
            # else:
            #     print("You could choose 'dqn', 'doubledqn' as network's parameter")
            #     return 1
            return 0


if __name__ == "__main__":
    sys.exit(main())
