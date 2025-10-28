from top import cchess_main
import argparse
from chessgame import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'play'], type=str, help='train or play')
    parser.add_argument('--ai_count', default=1, choices=[1, 2], type=int, help='choose ai player count')
    parser.add_argument('--ai_function', default='mcts', choices=['mcts', 'net'], type=str, help='mcts or net')
    parser.add_argument('--train_playout', default=400, type=int, help='mcts train playout')
    parser.add_argument('--batch_size', default=512, type=int, help='train batch_size')
    parser.add_argument('--play_playout', default=400, type=int, help='mcts play playout')
    parser.add_argument('--delay', dest='delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='Set how many seconds you want to delay after each move')
    parser.add_argument('--end_delay', dest='end_delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='Set how many seconds you want to delay after the end of game')
    parser.add_argument('--search_threads', default=16, type=int, help='search_threads')
    parser.add_argument("--processor", type=int, default=0, help="处理器编号")
    parser.add_argument('--num_gpus', default=1, type=int, help='gpu counts')
    parser.add_argument('--res_block_nums', default=7, type=int, help='res_block_nums')
    parser.add_argument('--human_color', default='b', choices=['w', 'b'], type=str, help='w or b')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)
        train_main.run()
    elif args.mode == 'play':
        game = chessgame(args.ai_count, args.ai_function, args.play_playout, args.delay, args.end_delay, args.batch_size,
                         args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)
        game.start()