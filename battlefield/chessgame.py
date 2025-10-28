from chessboard import *
from chessview import chessview
from top import cchess_main
import tkinter
import time


def real_coord(x):
    # 把鼠标点击的像素坐标坐标转换成棋盘上的格子坐标
    if x <= 50:
        return 0
    else:
        return (x - 50) // 40 + 1


def board_coord(x):
    # 把棋盘格子坐标转换成界面上的像素坐标
    return 30 + 40 * x


class chessgame:
    board = None  # 棋盘对象，后面会初始化
    cur_round = 1  # 当前回合数，从1开始
    game_mode = 1  # 游戏模式：0是人vs人，1是人vsAI，2是AIvsAI
    time_red = []  # 红方每步的思考时间
    time_green = []  # 绿方每步的思考时间

    def __init__(self, in_ai_count, in_ai_function, in_play_playout, in_delay, in_end_delay, batch_size, search_threads,
                 processor, num_gpus, res_block_nums, human_color="b"):
        # 初始化游戏参数
        self.human_color = human_color  # 人控制的颜色，默认是黑方（b）
        self.current_player = "w"  # 当前该谁走棋，默认先红方（w）
        self.players = {}  # 记录双方是人机还是AI
        self.players[self.human_color] = "human"  # 人控制的颜色标记为"human"
        # 算一下AI控制的颜色（和人相反）
        ai_color = "w" if self.human_color == "b" else "b"
        self.players[ai_color] = "AI"  # AI控制的颜色标记为"AI"

        # 初始化棋盘，参数决定上方是不是红方
        chessgame.board = chessboard(self.human_color == 'b')
        # 初始化界面，把当前游戏对象和棋盘传进去
        self.view = chessview(self, board=chessgame.board)
        self.view.showMsg("Loading Models...")  # 界面显示"加载模型中..."
        self.view.draw_board(self.board)  # 先画一下初始棋盘
        chessgame.game_mode = in_ai_count  # 设置游戏模式
        self.ai_function = in_ai_function  # AI决策函数
        self.play_playout = in_play_playout  # MCTS模拟次数
        self.delay = in_delay  # AI走棋延迟（秒）
        self.end_delay = in_end_delay  # 游戏结束后停留时间

        # 记录双方胜率估计
        self.win_rate = {}
        self.win_rate['w'] = 0.0
        self.win_rate['b'] = 0.0

        # 刷新一下界面，显示加载状态
        self.view.root.update()
        # 初始化象棋引擎（包含AI逻辑）
        self.cchess_engine = cchess_main(playout=self.play_playout, in_batch_size=batch_size, exploration=False,
                                         in_search_threads=search_threads,
                                         processor=processor, num_gpus=num_gpus, res_block_nums=res_block_nums,
                                         human_color=human_color)

    def player_is_red(self):
        # 判断当前走棋的是不是红方
        return self.current_player == "w"

    def start(self):
        # 开始游戏
        self.view.showMsg("Red")  # 界面标题显示"Red"（红方回合）
        if self.game_mode == 1:
            # 人vsAI模式：如果红方是AI，先让AI走第一步
            print('-----Round %d-----' % self.cur_round)
            if self.players["w"] == "AI":
                self.win_rate['w'] = self.perform_AI()  # AI走棋
                self.view.draw_board(self.board)  # 刷新棋盘
                self.change_player()  # 切换玩家
        elif self.game_mode == 2:
            # AIvsAI模式：直接开始第一轮
            print('-----Round %d-----' % self.cur_round)
            self.win_rate['w'] = self.perform_AI()
            self.view.draw_board(self.board)

        # 启动界面主循环
        self.view.start()

    def disp_mcts_msg(self):
        # 在界面上显示"MCTS正在搜索..."
        self.view.showMsg("MCTS Searching...")

    def callback(self, event):
        # 处理鼠标点击事件（人走棋时用）
        # 如果是人机模式且当前该AI走，直接返回不处理
        if self.game_mode == 1 and self.players[self.current_player] == "AI":
            return
        # AIvsAI模式下不处理鼠标点击
        if self.game_mode == 2:
            return

        # 把鼠标点击的像素坐标转成棋盘格子坐标
        rx, ry = real_coord(event.x), real_coord(event.y)
        # 调用棋盘的select方法处理选子和移动
        change, coord = self.board.select(rx, ry, self.player_is_red())

        # 如果之前有显示文字提示，清除一下
        if self.view.print_text_flag == True:
            self.view.print_text_flag = False
            self.view.can.create_image(0, 0, image=self.view.img, anchor=tkinter.NW)

        # 刷新棋盘显示
        self.view.draw_board(self.board)

        # 检查游戏是否结束
        if self.check_end():
            self.view.root.update()
            self.quit()  # 结束游戏
            return

        # 如果成功移动了棋子
        if change:
            # 第一回合且人是红方时，显示AI正在思考
            if self.cur_round == 1 and self.human_color == 'w':
                self.view.showMsg("MCTS Searching...")

            # 告诉引擎人走了哪步，获取AI的胜率估计
            self.win_rate[self.current_player] = self.cchess_engine.human_move(coord, self.ai_function)

            # 再次检查游戏是否结束
            if self.check_end():
                self.view.root.update()
                self.quit()
                return

            # 切换玩家，看是否需要AI走棋
            performed = self.change_player()
            if performed:
                self.view.draw_board(self.board)
                # 再检查一次游戏是否结束
                if self.check_end():
                    self.view.root.update()
                    self.quit()
                    return
                self.change_player()  # 再切一次（因为AI走了一步）

    def quit(self):
        # 游戏结束后停留指定时间，然后退出
        time.sleep(self.end_delay)
        self.view.quit()

    def check_end(self):
        # 调用引擎检查游戏是否结束
        ret, winner = self.cchess_engine.check_end()
        if ret == True:
            # 根据赢家显示不同信息
            if winner == "b":
                self.view.showMsg('*****绿方赢了！回合：%d*****' % self.cur_round)
            elif winner == "w":
                self.view.showMsg('*****红方赢了！回合：%d*****' % self.cur_round)
            elif winner == "t":
                self.view.showMsg('*****平局！回合：%d*****' % self.cur_round)
            self.view.root.update()  # 刷新界面显示结果
        return ret

    def _check_end(self, board):
        # 备用的检查结束方法（通过判断将帅是否存在）
        red_king = False  # 红方帅是否还在
        green_king = False  # 绿方将是否还在
        pieces = board.pieces
        for (x, y) in pieces.keys():
            if pieces[x, y].is_king:  # 如果是将帅
                if pieces[x, y].is_red:
                    red_king = True
                else:
                    green_king = True
        # 红方帅没了，绿方赢
        if not red_king:
            self.view.showMsg('*****绿方赢了！回合：%d*****' % self.cur_round)
            self.view.root.update()
            return True
        # 绿方将没了，红方赢
        elif not green_king:
            self.view.showMsg('*****红方赢了！回合：%d*****' % self.cur_round)
            self.view.root.update()
            return True
        # 超过200回合判平局
        elif self.cur_round >= 200:
            self.view.showMsg('*****平局！回合：%d*****' % self.cur_round)
            self.view.root.update()
            return True
        return False

    def change_player(self):
        # 切换当前玩家（红变绿，绿变红）
        self.current_player = "w" if self.current_player == "b" else "b"

        # 如果切换到红方，说明回合数加1
        if self.current_player == "w":
            self.cur_round += 1
            print('-----Round %d-----' % self.cur_round)

        # 准备显示双方的胜率估计
        red_msg = " ({:.4f})".format(self.win_rate['w'])
        green_msg = " ({:.4f})".format(self.win_rate['b'])

        # 获取AI推荐的走法和概率，显示在列表里
        sorted_move_probs = self.cchess_engine.get_hint(self.ai_function, True, self.disp_mcts_msg)
        self.view.print_all_hint(sorted_move_probs)

        # 更新界面标题，显示当前轮到谁走，以及双方胜率
        if self.current_player == "w":
            self.view.showMsg("红方" + red_msg + " 绿方" + green_msg)
        else:
            self.view.showMsg("绿方" + green_msg + " 红方" + red_msg)
        self.view.root.update()  # 刷新界面

        # 根据游戏模式判断是否需要AI走棋
        if self.game_mode == 1:
            # 人vsAI：如果当前是AI的回合，就让AI走
            if self.players[self.current_player] == "AI":
                self.win_rate[self.current_player] = self.perform_AI()
                return True
            return False
        elif self.game_mode == 2:
            # AIvsAI：直接让AI走
            self.win_rate[self.current_player] = self.perform_AI()
            return True
        return False

    def perform_AI(self):
        # 让AI走一步棋
        print('...AI正在思考...')
        START_TIME = time.perf_counter()  # 记录开始时间
        # 调用引擎让AI选一步棋，返回走法和胜率
        move, win_rate = self.cchess_engine.select_move(self.ai_function)
        time_used = time.perf_counter() - START_TIME  # 计算用时
        print('...用时%fs...' % time_used)

        # 记录该步的思考时间（红方或绿方）
        if self.current_player == "w":
            self.time_red.append(time_used)
        else:
            self.time_green.append(time_used)

        # 如果有走法，就在棋盘上执行移动
        if move is not None:
            self.board.move(move[0], move[1], move[2], move[3])

        return win_rate  # 返回AI对这步棋的胜率估计

    # AI VS AI模式专用：控制双方轮流走棋
    def game_mode_2(self):
        self.change_player()  # 切换玩家
        self.view.draw_board(self.board)  # 刷新棋盘
        self.view.root.update()
        # 检查游戏是否结束，结束就返回True
        if self.check_end():
            return True
        return False