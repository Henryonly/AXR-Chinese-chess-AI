from asyncio import Future
import asyncio
from asyncio.queues import Queue
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用OneDNN优化（避免某些兼容性问题）
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x特性，使用1.x模式
import numpy as np
import os
import sys
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
import copy
from policy_value_network import *  # 导入策略价值网络
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

def flipped_uci_labels(param):
    """翻转UCI走法标签（比如把黑方视角的走法转成红方视角）"""
    def repl(x):
        # 数字部分翻转（0和9换，1和8换...），字母不变
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])
    return [repl(x) for x in param]

# 创建所有合法走子的UCI标签（共2086种）
def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']  # 横坐标（9列）
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 纵坐标（10行）

    # 士的特殊走法（九宫格内斜着走）
    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    # 相的特殊走法（田字格，不能过河）
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    # 车、马、炮等的基本走法（车横竖走，马走日等）
    letters = 'abcdefghi'  # 9 列
    numbers = '0123456789'  # 10 行

    labels_array = []

    for l1 in range(9):  # 起始横坐标
        for n1 in range(10):  # 起始纵坐标
            # 车的走法：横、竖全线
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)]
            # 马的走法：日字形 8 个方向
            knight_deltas = [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                             (2, -1), (-1, 2), (2, 1), (1, 2)]
            destinations += [(l1 + dx, n1 + dy) for dx, dy in knight_deltas]

            # 生成走法标签
            for l2, n2 in destinations:
                if (l1, n1) != (l2, n2) and 0 <= l2 < 9 and 0 <= n2 < 10:
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)

    # 加上士和相的特殊走法
    for p in Advisor_labels:
        labels_array.append(p)
    for p in Bishop_labels:
        labels_array.append(p)

    return labels_array

# 创建棋盘位置标签（用于坐标转换）
def create_position_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()  # 反转字母顺序
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]
            labels_array.append(move)
    return labels_array

# 创建反转的棋盘位置标签
def create_position_labels_reverse():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()  # 反转整个列表
    return labels_array

class leaf_node(object):
    """MCTS中的叶子节点类（每个节点代表一个棋盘状态和可能的走法）"""
    def __init__(self, in_parent, in_prior_p, in_state):
        self.P = in_prior_p  # 先验概率（来自神经网络）
        self.Q = 0  # 行动价值（平均赢率）
        self.N = 0  # 访问次数
        self.v = 0  # 节点价值（神经网络评估）
        self.U = 0  # 探索项（UCB公式中的部分）
        self.W = 0  # 总价值（用于计算Q）
        self.parent = in_parent  # 父节点
        self.child = {}  # 子节点（key是走法，value是节点）
        self.state = in_state  # 当前节点的棋盘状态

    def is_leaf(self):
        """判断是否为叶子节点（没有子节点）"""
        return self.child == {}

    def get_Q_plus_U_new(self, c_puct):
        """计算Q+U（新的计算方式，用于选择子节点）"""
        U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + U

    def get_Q_plus_U(self, c_puct):
        """计算Q+U（更新U的值，用于选择子节点）"""
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + self.U

    def select_new(self, c_puct):
        """选择Q+U最大的子节点（新方式）"""
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U_new(c_puct))

    def select(self, c_puct):
        """选择Q+U最大的子节点（旧方式）"""
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))

    def expand(self, moves, action_probs):
        """扩展节点：为所有合法走法创建子节点"""
        tot_p = 1e-8  # 防止除零
        action_probs = action_probs.flatten()  # 展平概率数组
        for action in moves:
            # 模拟走这步棋后的新状态
            in_state = GameBoard.sim_do_action(action, self.state)
            # 获取该走法的先验概率
            mov_p = action_probs[label2i[action]]
            # 创建子节点
            new_node = leaf_node(self, mov_p, in_state)
            self.child[action] = new_node
            tot_p += mov_p  # 累计概率

        # 归一化先验概率（确保和为1）
        for a, n in self.child.items():
            n.P /= tot_p

    def back_up_value(self, value):
        """反向传播价值（更新当前节点的统计信息）"""
        self.N += 1  # 访问次数+1
        self.W += value  # 总价值累加
        self.v = value  # 记录当前价值
        self.Q = self.W / self.N  # 更新平均价值
        # 更新探索项
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / (1 + self.N)

    def backup(self, value):
        """从当前节点向上反向传播价值（更新所有祖先节点）"""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.v = value
            node.Q = node.W / node.N  # 平均价值
            node = node.parent  # 移到父节点
            value = -value  # 价值翻转（因为双方轮流走）

# 棋子类型映射（用于将棋盘状态转成神经网络输入）
pieces_order = 'KARBNPCkarbnpc'  # 14种棋子（红方KARBNPC，黑方karbnpc）
ind = {pieces_order[i]: i for i in range(14)}  # 每种棋子对应一个索引

# 生成所有走法标签和映射关系
labels_array = create_uci_labels()
labels_len = len(labels_array)
flipped_labels = flipped_uci_labels(labels_array)  # 翻转后的标签（用于黑方视角）
unflipped_index = [labels_array.index(x) for x in flipped_labels]  # 记录翻转前后的索引对应关系

# 走法和索引的双向映射
i2label = {i: val for i, val in enumerate(labels_array)}
label2i = {val: i for i, val in enumerate(labels_array)}

def get_pieces_count(state):
    """统计棋盘上的棋子数量（用于判断是否吃子）"""
    count = 0
    for s in state:
        if s.isalpha():  # 字母表示棋子
            count += 1
    return count

def is_kill_move(state_prev, state_next):
    """判断是否是吃子走法（前后棋子数量差）"""
    return get_pieces_count(state_prev) - get_pieces_count(state_next)

# 用于异步预测的队列项（包含特征和未来结果）
QueueItem = namedtuple("QueueItem", "feature future")
c_PUCT = 5  # MCTS中的探索系数
virtual_loss = 3  # 虚拟损失（用于多线程同步）
cut_off_depth = 30  # 搜索深度限制

class MCTS_tree(object):
    """蒙特卡洛树搜索（MCTS）类"""
    def __init__(self, in_state, in_forward, search_threads):
        self.noise_eps = 0.25  # 探索噪声比例
        self.dirichlet_alpha = 0.3  # 狄利克雷分布参数（用于增加探索）
        # 根节点的先验概率（带探索噪声）
        self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
        self.root = leaf_node(None, self.p_, in_state)  # 根节点
        self.c_puct = 5  # 探索系数（UCB公式中的参数）
        self.forward = in_forward  # 神经网络的前向传播函数
        self.node_lock = defaultdict(Lock)  # 节点锁（多线程安全）

        self.virtual_loss = 3  # 虚拟损失值
        self.now_expanding = set()  # 正在扩展的节点集合
        self.expanded = set()  # 已扩展的节点集合
        self.cut_off_depth = 30  # 搜索深度限制
        self.sem = asyncio.Semaphore(search_threads)  # 控制并发搜索的数量
        self.queue = Queue(search_threads)  # 异步预测队列
        self.loop = asyncio.get_event_loop()  # 事件循环（用于异步操作）
        self.running_simulation_num = 0  # 正在运行的模拟数量

    def reload(self):
        """重置MCTS树（回到初始状态）"""
        self.root = leaf_node(None, self.p_,
                         "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")  # 初始棋盘
        self.expanded = set()

    def Q(self, move) -> float:
        """获取某个走法的行动价值Q"""
        ret = 0.0
        find = False
        for a, n in self.root.child.items():
            if move == a:
                ret = n.Q
                find = True
        if not find:
            print(f"{move} 不在子节点中")
        return ret

    def update_tree(self, act):
        """根据走法更新树（将对应子节点设为新根）"""
        self.expanded.discard(self.root)  # 移除旧根的扩展标记
        self.root = self.root.child[act]  # 新根设为走法对应的子节点
        self.root.parent = None  # 新根没有父节点

    def is_expanded(self, key) -> bool:
        """判断节点是否已扩展"""
        return key in self.expanded

    async def tree_search(self, node, current_player, restrict_round) -> float:
        """异步执行树搜索（多线程并行）"""
        self.running_simulation_num += 1  # 增加正在运行的模拟数量

        async with self.sem:  # 控制并发数
            value = await self.start_tree_search(node, current_player, restrict_round)
            self.running_simulation_num -= 1  # 减少正在运行的模拟数量
            return value

    async def start_tree_search(self, node, current_player, restrict_round)->float:
        """开始树搜索（递归过程）"""
        now_expanding = self.now_expanding

        # 如果节点正在扩展，等待一会儿
        while node in now_expanding:
            await asyncio.sleep(1e-4)

        # 如果节点未扩展，调用神经网络评估并扩展
        if not self.is_expanded(node):
            self.now_expanding.add(node)  # 标记为正在扩展

            # 生成神经网络输入
            positions = self.generate_inputs(node.state, current_player)
            # 加入预测队列并等待结果
            future = await self.push_queue(positions)
            await future
            action_probs, value = future.result()  # 得到策略概率和价值

            # 如果是黑方回合，翻转策略概率（适应红方视角的网络）
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)

            # 获取当前状态的合法走法
            moves = GameBoard.get_legal_moves(node.state, current_player)
            # 扩展节点（创建子节点）
            node.expand(moves, action_probs)
            self.expanded.add(node)  # 标记为已扩展

            self.now_expanding.remove(node)  # 移除正在扩展标记
            return value[0] * -1  # 返回价值（翻转，因为是对手视角）

        else:
            # 节点已扩展，选择子节点继续搜索
            last_state = node.state
            action, node = node.select_new(c_PUCT)  # 选择Q+U最大的子节点
            # 切换当前玩家
            current_player = "w" if current_player == "b" else "b"
            # 更新限制回合数（无吃子的连续回合）
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

            # 增加虚拟损失（防止多线程重复选择同一节点）
            node.N += virtual_loss
            node.W += -virtual_loss

            # 检查游戏是否结束（将帅被吃）
            if (node.state.find('K') == -1 or node.state.find('k') == -1):
                if (node.state.find('K') == -1):  # 红帅被吃
                    value = 1.0 if current_player == "b" else -1.0
                if (node.state.find('k') == -1):  # 黑将被吃
                    value = -1.0 if current_player == "b" else 1.0
                value = value * -1  # 翻转价值
            # 检查是否平局（60回合无吃子）
            elif restrict_round >= 60:
                value = 0.0
            else:
                # 递归搜索子节点
                value = await self.start_tree_search(node, current_player, restrict_round)

            # 撤销虚拟损失
            node.N += -virtual_loss
            node.W += virtual_loss
            # 反向传播价值
            node.back_up_value(value)
            return value * -1  # 翻转价值（因为是对手视角）

    async def prediction_worker(self):
        """异步预测工作线程（批量处理神经网络推理）"""
        q = self.queue
        margin = 10  # 等待时间余量（避免提前结束）
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)  # 没任务时休眠
                continue
            # 一次性取出队列中所有任务
            item_list = [q.get_nowait() for _ in range(q.qsize())]
            # 批量处理特征
            features = np.asarray([item.feature for item in item_list])
            action_probs, value = self.forward(features)  # 调用神经网络
            # 分配结果给每个任务
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        """将特征加入预测队列，返回未来结果"""
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)  # 加入队列
        return future

    def main(self, state, current_player, restrict_round, playouts):
        """执行MCTS主循环（进行指定次数的模拟）"""
        node = self.root
        # 如果根节点未扩展，先扩展它
        if not self.is_expanded(node):
            positions = self.generate_inputs(node.state, current_player)
            positions = np.expand_dims(positions, 0)  # 增加批次维度
            action_probs, value = self.forward(positions)  # 神经网络评估
            # 黑方回合则翻转策略
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            # 获取合法走法并扩展
            moves = GameBoard.get_legal_moves(node.state, current_player)
            node.expand(moves, action_probs)
            self.expanded.add(node)

        # 创建异步任务列表
        coroutine_list = []
        for _ in range(playouts):
            coroutine_list.append(self.tree_search(node, current_player, restrict_round))
        coroutine_list.append(self.prediction_worker())  # 加入预测工作线程
        # 运行所有异步任务
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

    def do_simulation(self, state, current_player, restrict_round):
        """执行单次模拟（旧版本，用于对比）"""
        node = self.root
        last_state = state
        # 沿着树搜索到叶子节点
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            current_player = "w" if current_player == "b" else "b"
            # 更新限制回合数
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

        # 评估叶子节点
        positions = self.generate_inputs(node.state, current_player)
        positions = np.expand_dims(positions, 0)
        action_probs, value = self.forward(positions)
        # 黑方回合则翻转策略
        if self.is_black_turn(current_player):
            action_probs = cchess_main.flip_policy(action_probs)

        # 检查游戏是否结束
        if (node.state.find('K') == -1 or node.state.find('k') == -1):
            if (node.state.find('K') == -1):
                value = 1.0 if current_player == "b" else -1.0
            if (node.state.find('k') == -1):
                value = -1.0 if current_player == "b" else 1.0
        elif restrict_round >= 60:
            value = 0.0
        else:
            # 扩展叶子节点
            moves = GameBoard.get_legal_moves(node.state, current_player)
            node.expand(moves, action_probs)

        # 反向传播价值
        node.backup(-value)

    def generate_inputs(self, in_state, current_player):
        """生成神经网络的输入特征"""
        # 根据当前玩家决定是否翻转棋盘（统一用红方视角输入网络）
        state, palyer = self.try_flip(in_state, current_player, self.is_black_turn(current_player))
        return self.state_to_positions(state)

    def replace_board_tags(self, board):
        """替换棋盘表示中的数字（将1-9替换为对应数量的1，方便解析）"""
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.replace("/", "")  # 去掉分隔符

    def state_to_positions(self, state):
        """将棋盘状态转换为神经网络输入（9x10x14的特征图）"""
        board_state = self.replace_board_tags(state)
        # 初始化特征图（9列x10行x14通道，每个通道对应一种棋子）
        pieces_plane = np.zeros(shape=(9, 10, 14), dtype=np.float32)
        for rank in range(9):  # 列
            for file in range(10):  # 行
                # 获取当前位置的字符
                v = board_state[rank * 9 + file]
                if v.isalpha():  # 如果是棋子
                    pieces_plane[rank][file][ind[v]] = 1  # 在对应通道标记1
        assert pieces_plane.shape == (9, 10, 14)  # 确保形状正确
        return pieces_plane

    def try_flip(self, state, current_player, flip=False):
        """翻转棋盘（黑方视角转红方视角，统一网络输入）"""
        if not flip:
            return state, current_player

        rows = state.split('/')  # 按行分割

        def swapcase(a):
            """翻转字母大小写（红黑棋子互换）"""
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a

        def swapall(aa):
            """对一行的所有字符翻转大小写"""
            return "".join([swapcase(a) for a in aa])

        # 翻转行顺序并交换大小写（黑方视角转红方）
        return "/".join([swapall(row) for row in reversed(rows)]),  ('w' if current_player == 'b' else 'b')

    def is_black_turn(self, current_player):
        """判断当前是否是黑方回合"""
        return current_player == 'b'

class GameBoard(object):
    """棋盘类（处理棋盘状态、走法生成等）"""
    # 棋盘位置名称（用于坐标转换）
    board_pos_name = np.array(create_position_labels()).reshape(9,10).transpose()
    Ny = 10  # 行数
    Nx = 9   # 列数

    def __init__(self):
        # 初始棋盘状态（FEN格式）
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"
        self.round = 1  # 当前回合数
        self.current_player = "w"  # 当前玩家（w:红方，b:黑方）
        self.restrict_round = 0  # 无吃子的连续回合数

    def reload(self):
        """重置棋盘到初始状态"""
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board, action = None):
        """打印棋盘（调试用）"""
        def string_reverse(string):
            return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if action is not None:
            src = action[0:2]
            src_x = int(x_trans[src[0]])
            src_y = int(src[1])

        # 替换数字为空格（方便查看）
        board = board.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')  # 按行分割
        print("  abcdefghi")  # 列标
        for i, line in enumerate(board):
            if action is not None and i == src_y:
                # 标记走法的起点
                s = list(line)
                s[src_x] = 'x'
                line = ''.join(s)
            print(i, line)  # 打印行号和内容

    @staticmethod
    def sim_do_action(in_action, in_state):
        """模拟执行走法，返回新的棋盘状态"""
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

        # 解析走法（前两位是起点，后两位是终点）
        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])
        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # 将棋盘转换为可修改的列表形式
        board_positions = GameBoard.board_to_pos_name(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)

        # 执行移动：将起点棋子移到终点，起点置为1（空）
        lines[dst_y][dst_x] = lines[src_y][src_x]
        lines[src_y][src_x] = '1'

        # 转换回字符串形式
        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])

        # 将连续的1替换为数字（压缩表示）
        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")

        return board

    @staticmethod
    def board_to_pos_name(board):
        """将棋盘的压缩表示转换为展开形式（数字转多个1）"""
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")  # 按行分割

    @staticmethod
    def check_bounds(toY, toX):
        """检查坐标是否在棋盘范围内"""
        if toY < 0 or toX < 0:
            return False
        if toY >= GameBoard.Ny or toX >= GameBoard.Nx:
            return False
        return True

    @staticmethod
    def validate_move(c, upper=True):
        """验证移动是否合法（是否吃己方棋子）"""
        if c.isalpha():  # 是棋子
            if upper:  # 当前是红方回合，不能吃红子
                return c.islower()  # 只能吃黑子（小写）
            else:  # 当前是黑方回合，不能吃黑子
                return c.isupper()  # 只能吃红子（大写）
        else:  # 空位，合法
            return True

    @staticmethod
    def get_legal_moves(state, current_player):
        """生成当前状态下的所有合法走法"""
        moves = []
        k_x = None  # 黑将x坐标
        k_y = None  # 黑将y坐标
        K_x = None  # 红帅x坐标
        K_y = None  # 红帅y坐标
        face_to_face = False  # 是否将帅对面（可吃）

        # 解析棋盘状态
        board_positions = np.array(GameBoard.board_to_pos_name(state))
        for y in range(board_positions.shape[0]):  # 遍历行
            for x in range(len(board_positions[y])):  # 遍历列
                c = board_positions[y][x]
                if c.isalpha():  # 如果是棋子
                    # 黑车（r）的走法（当前是黑方回合）
                    if c == 'r' and current_player == 'b':
                        # 向左走
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():  # 遇到红子，可吃
                                    moves.append(m)
                                break  # 遇到棋子就停止
                            moves.append(m)  # 空位，加入走法
                        # 向右走
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break
                            moves.append(m)
                        # 向上走
                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break
                            moves.append(m)
                        # 向下走
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].isupper():
                                    moves.append(m)
                                break
                            moves.append(m)

                    # 红车（R）的走法（当前是红方回合）
                    elif c == 'R' and current_player == 'w':
                        # 类似黑车，略...
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break
                            moves.append(m)
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break
                            moves.append(m)
                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break
                            moves.append(m)
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if board_positions[toY][toX].isalpha():
                                if board_positions[toY][toX].islower():
                                    moves.append(m)
                                break
                            moves.append(m)

                    # 黑马（n/h）的走法（当前是黑方回合）
                    elif (c == 'n' or c == 'h') and current_player == 'b':
                        # 马走日，8个方向，需检查绊马腿
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                # 先横跳2，再竖跳1
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if (GameBoard.check_bounds(toY, toX) and
                                    GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                    not board_positions[toY - i][x].isalpha()):  # 无绊马腿
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                # 先竖跳1，再横跳2
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if (GameBoard.check_bounds(toY, toX) and
                                    GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                    not board_positions[y][toX - j].isalpha()):  # 无绊马腿
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 红马（N/H）的走法（当前是红方回合）
                    elif (c == 'N' or c == 'H') and current_player == 'w':
                        # 类似黑马，略...
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if (GameBoard.check_bounds(toY, toX) and
                                    GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                    not board_positions[toY - i][x].isalpha()):
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if (GameBoard.check_bounds(toY, toX) and
                                    GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                    not board_positions[y][toX - j].isalpha()):
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 黑相（b/e）的走法（当前是黑方回合，不能过河）
                    elif (c == 'b' or c == 'e') and current_player == 'b':
                        # 相走田，4个方向，需检查绊相腿，且不能过河（y>=5）
                        for i in range(-2, 3, 4):  # i=±2
                            toY = y + i
                            toX = x + i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                toY >= 5 and  # 不能过河
                                not board_positions[y + i//2][x + i//2].isalpha()):  # 无绊相腿
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                toY >= 5 and
                                not board_positions[y + i//2][x - i//2].isalpha()):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 红相（B/E）的走法（当前是红方回合，不能过河）
                    elif (c == 'B' or c == 'E') and current_player == 'w':
                        # 类似黑相，略（不能过河y<=4）
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                toY <= 4 and
                                not board_positions[y + i//2][x + i//2].isalpha()):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                toY <= 4 and
                                not board_positions[y + i//2][x - i//2].isalpha()):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 黑士（a）的走法（当前是黑方回合，九宫格内）
                    elif c == 'a' and current_player == 'b':
                        # 士走斜线，4个方向，必须在九宫格（y>=7, x3-5）
                        for i in range(-1, 3, 2):  # i=±1
                            toY = y + i
                            toX = x + i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                toY >=7 and toX >=3 and toX <=5):  # 九宫格内
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                toY >=7 and toX >=3 and toX <=5):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 红士（A）的走法（当前是红方回合，九宫格内）
                    elif c == 'A' and current_player == 'w':
                        # 类似黑士，略（九宫格y<=2, x3-5）
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                toY <=2 and toX >=3 and toX <=5):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i
                            if (GameBoard.check_bounds(toY, toX) and
                                GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                toY <=2 and toX >=3 and toX <=5):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 黑将（k）的走法（当前是黑方回合，九宫格内）
                    elif c == 'k':
                        k_x = x
                        k_y = y
                        if current_player == 'b':
                            # 将走一步，4个方向，九宫格内（y>=7, x3-5）
                            for i in range(2):
                                for sign in range(-1, 2, 2):  # sign=±1
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign
                                    if (GameBoard.check_bounds(toY, toX) and
                                        GameBoard.validate_move(board_positions[toY][toX], upper=False) and
                                        toY >=7 and toX >=3 and toX <=5):
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 红帅（K）的走法（当前是红方回合，九宫格内）
                    elif c == 'K':
                        K_x = x
                        K_y = y
                        if current_player == 'w':
                            # 类似黑将，略（九宫格y<=2, x3-5）
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign
                                    if (GameBoard.check_bounds(toY, toX) and
                                        GameBoard.validate_move(board_positions[toY][toX], upper=True) and
                                        toY <=2 and toX >=3 and toX <=5):
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 黑炮（c）的走法（当前是黑方回合）
                    elif c == 'c' and current_player == 'b':
                        # 炮走直线，吃子时需有炮架
                        toY = y
                        hits = False  # 是否遇到炮架
                        # 向左走
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True  # 遇到第一个棋子，作为炮架
                                else:
                                    moves.append(m)  # 空位，可走
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():  # 遇到红子，可吃
                                        moves.append(m)
                                    break  # 吃子后停止
                        # 向右走（类似向左）
                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break
                        # 向上走（类似向左）
                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break
                        # 向下走（类似向左）
                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].isupper():
                                        moves.append(m)
                                    break

                    # 红炮（C）的走法（当前是红方回合）
                    elif c == 'C' and current_player == 'w':
                        # 类似黑炮，略（吃黑子）
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break
                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break
                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break
                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if not hits:
                                if board_positions[toY][toX].isalpha():
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if board_positions[toY][toX].isalpha():
                                    if board_positions[toY][toX].islower():
                                        moves.append(m)
                                    break

                    # 黑兵（p）的走法（当前是黑方回合）
                    elif c == 'p' and current_player == 'b':
                        # 未过河只能前进，过河后可左右走
                        toY = y - 1  # 向上走（黑兵向前）
                        toX = x
                        if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                        # 过河后（y < 5）可左右走
                        if y < 5:
                            toY = y
                            toX = x + 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toX = x - 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    # 红兵（P）的走法（当前是红方回合）
                    elif c == 'P' and current_player == 'w':
                        # 未过河只能前进，过河后可左右走
                        toY = y + 1  # 向下走（红兵向前）
                        toX = x
                        if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                        # 过河后（y > 4）可左右走
                        if y > 4:
                            toY = y
                            toX = x + 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toX = x - 1
                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

        # 检查将帅是否对面（中间无棋子），可直接吃
        if K_x is not None and k_x is not None and K_x == k_x:
            face_to_face = True
            # 检查中间是否有棋子
            for i in range(K_y + 1, k_y, 1):
                if board_positions[i][K_x].isalpha():
                    face_to_face = False
                    break

        # 如果将帅对面，加入吃将/帅的走法
        if face_to_face:
            if current_player == 'b':
                moves.append(GameBoard.board_pos_name[k_y][k_x] + GameBoard.board_pos_name[K_y][K_x])
            else:
                moves.append(GameBoard.board_pos_name[K_y][K_x] + GameBoard.board_pos_name[k_y][k_x])

        return moves

def softmax(x):
    """softmax函数（将数值转为概率分布）"""
    probs = np.exp(x - np.max(x))  # 减最大值防止溢出
    probs /= np.sum(probs)  # 归一化
    return probs

class cchess_main(object):
    """象棋主类（整合MCTS、神经网络和棋盘逻辑）"""
    def __init__(self, playout=400, in_batch_size=128, exploration=True, in_search_threads=16,
                 processor="cpu", num_gpus=1, res_block_nums=7, human_color='b'):
        self.epochs = 5  # 每次策略更新的训练轮数
        self.playout_counts = playout  # MCTS模拟次数
        self.temperature = 1  # 温度参数（控制探索程度）
        self.batch_size = in_batch_size  # 训练批次大小
        self.game_batch = 400  # 每多少局更新一次策略
        self.top_steps = 30  # 前多少步使用较高温度
        self.top_temperature = 1  # 前几步的温度
        self.eta = 0.03  # 学习率衰减参数
        self.learning_rate = 0.001  # 初始学习率
        self.lr_multiplier = 1.0  # 学习率乘数（动态调整）
        self.buffer_size = 10000  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)  # 经验池（存储自我对弈数据）
        self.game_borad = GameBoard()  # 棋盘对象
        # 初始化策略价值网络
        self.policy_value_netowrk = policy_value_network(res_block_nums)
        self.search_threads = in_search_threads  # MCTS搜索线程数
        # 初始化MCTS树
        self.mcts = MCTS_tree(self.game_borad.state, self.policy_value_netowrk.forward, self.search_threads)
        self.exploration = exploration  # 是否启用探索
        self.resign_threshold = -0.8  # 认输阈值
        self.global_step = 0  # 全局训练步数
        self.kl_targ = 0.025  # KL散度目标（控制策略更新幅度）
        # 日志文件
        self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
        self.human_color = human_color  # 人控制的颜色

    @staticmethod
    def flip_policy(prob):
        """翻转策略概率（黑方视角转红方视角）"""
        prob = prob.flatten()
        return np.asarray([prob[ind] for ind in unflipped_index])

    def policy_update(self):
        """更新策略网络（从经验池中采样训练）"""
        # 从经验池随机采样
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.expand_dims(winner_batch, 1)  # 增加维度

        start_time = time.time()
        # 记录更新前的策略和价值（用于计算KL散度）
        old_probs, old_v = self.mcts.forward(state_batch)
        # 多轮训练
        for i in range(self.epochs):
            # 执行一次训练
            accuracy, loss, self.global_step = self.policy_value_netowrk.train_step(
                state_batch, mcts_probs_batch, winner_batch, self.learning_rate * self.lr_multiplier)
            # 计算更新后的策略和价值
            new_probs, new_v = self.mcts.forward(state_batch)
            # 计算KL散度（衡量策略变化大小）
            kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))
            kl_lst = []
            for line in kl_tmp:
                # 过滤异常值
                all_value = [x for x in line if str(x) != 'nan' and str(x) != 'inf']
                kl_lst.append(np.sum(all_value))
            kl = np.mean(kl_lst)
            # 如果KL散度过大，提前停止本轮训练
            if kl > self.kl_targ * 4:
                break
        # 保存模型
        self.policy_value_netowrk.save(self.global_step)
        print(f"训练用时 {time.time() - start_time} 秒")

        # 根据KL散度调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5  # 调小学习率
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5  # 调大学习率

        # 计算价值函数的解释方差（衡量价值预测好坏）
        explained_var_old = 1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch))
        # 打印并记录日志
        log_str = f"kl:{kl:.5f},lr_multiplier:{self.lr_multiplier:.3f},loss:{loss},accuracy:{accuracy},explained_var_old:{explained_var_old:.3f},explained_var_new:{explained_var_new:.3f}"
        print(log_str)
        self.log_file.write(log_str + '\n')
        self.log_file.flush()

    def run(self):
        """运行自我对弈并训练模型"""
        batch_iter = 0
        try:
            while True:
                batch_iter += 1
                # 进行一次自我对弈，获取数据
                play_data, episode_len = self.selfplay()
                print(f"批次 {batch_iter}, 对局长度: {episode_len}")
                # 处理数据并加入经验池
                extend_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                # 经验池足够大时，更新策略
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
        except KeyboardInterrupt:
            # 捕获中断信号，保存日志和模型
            self.log_file.close()
            self.policy_value_netowrk.save(self.global_step)

    def get_hint(self, mcts_or_net, reverse, disp_mcts_msg_handler):
        """获取走法提示（基于MCTS或直接用网络）"""
        if mcts_or_net == "mcts":
            # 用MCTS计算提示
            if self.mcts.root.child == {}:
                disp_mcts_msg_handler()  # 显示"搜索中..."
                # 执行MCTS搜索
                self.mcts.main(self.game_borad.state, self.game_borad.current_player,
                               self.game_borad.restrict_round, self.playout_counts)
            # 获取所有走法的访问次数
            actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
            actions, visits = zip(*actions_visits)
            # 计算概率（基于访问次数的softmax）
            probs = softmax(1.0 / self.temperature * np.log(visits))

            # 构建走法-概率字典（根据人控制的颜色调整走法标签）
            act_prob_dict = defaultdict(float)
            for i in range(len(actions)):
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(actions[i]))
                else:
                    action = actions[i]
                act_prob_dict[action] = probs[i]

        elif mcts_or_net == "net":
            # 直接用网络输出作为提示
            # 生成网络输入
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)

            # 黑方回合则翻转策略
            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            # 获取合法走法
            moves = GameBoard.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            # 计算合法走法的概率（归一化）
            tot_p = 1e-8
            action_probs = action_probs.flatten()
            act_prob_dict = defaultdict(float)
            for action in moves:
                mov_p = action_probs[label2i[action]]
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(action))
                act_prob_dict[action] = mov_p
                tot_p += mov_p
            # 归一化
            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p

        # 按概率排序
        sorted_move_probs = sorted(act_prob_dict.items(), key=lambda item: item[1], reverse=reverse)
        return sorted_move_probs

    def get_action(self, state, temperature=1e-3):
        """通过MCTS获取走法"""
        # 执行MCTS搜索
        self.mcts.main(state, self.game_borad.current_player, self.game_borad.restrict_round, self.playout_counts)
        # 获取所有走法的访问次数
        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        # 计算概率（基于访问次数）
        probs = softmax(1.0 / temperature * np.log(visits))
        move_probs = []
        move_probs.append([actions, probs])

        # 选择走法（带探索或纯贪心）
        if self.exploration:
            # 混合策略（75%基于概率，25%随机探索）
            act = np.random.choice(actions, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
        else:
            # 贪心选择概率最大的走法
            act = np.random.choice(actions, p=probs)

        # 获取该走法的赢率
        win_rate = self.mcts.Q(act)
        # 更新MCTS树（将选择的子节点设为新根）
        self.mcts.update_tree(act)

        return act, move_probs, win_rate

    def get_action_old(self, state, temperature=1e-3):
        """旧版本的获取走法（单次模拟）"""
        # 执行指定次数的模拟
        for i in range(self.playout_counts):
            state_sim = copy.deepcopy(state)
            self.mcts.do_simulation(state_sim, self.game_borad.current_player, self.game_borad.restrict_round)
        # 后续逻辑同get_action（略）
        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0 / temperature * np.log(visits))
        move_probs = []
        move_probs.append([actions, probs])

        if self.exploration:
            act = np.random.choice(actions, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        self.mcts.update_tree(act)
        return act, move_probs

    def check_end(self):
        """检查游戏是否结束"""
        # 检查将帅是否被吃
        if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
            if self.game_borad.state.find('K') == -1:  # 红帅被吃，黑方赢
                print("绿方获胜")
                return True, "b"
            if self.game_borad.state.find('k') == -1:  # 黑将被吃，红方赢
                print("红方获胜")
                return True, "w"
        # 检查是否平局（60回合无吃子）
        elif self.game_borad.restrict_round >= 60:
            print("平局！没有赢家！")
            return True, "t"
        else:
            return False, ""

    def human_move(self, coord, mcts_or_net):
        """处理人类走棋"""
        win_rate = 0
        x_trans = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i'}

        # 解析人类输入的坐标（src和dst）
        src = coord[0:2]
        dst = coord[2:4]
        src_x = (x_trans[src[0]])
        src_y = str(src[1])
        dst_x = (x_trans[dst[0]])
        dst_y = str(dst[1])
        action = src_x + src_y + dst_x + dst_y  # 生成走法标签

        # 根据人控制的颜色调整走法标签
        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))

        # 如果用MCTS，更新树并获取赢率
        if mcts_or_net == "mcts":
            if self.mcts.root.child == {}:
                self.mcts.main(self.game_borad.state, self.game_borad.current_player,
                               self.game_borad.restrict_round, self.playout_counts)
            win_rate = self.mcts.Q(action)
            self.mcts.update_tree(action)

        # 更新棋盘状态
        last_state = self.game_borad.state
        self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        # 切换玩家
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        # 更新无吃子回合数
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        return win_rate

    def select_move(self, mcts_or_net):
        """AI选择走法并执行"""
        if mcts_or_net == "mcts":
            # 用MCTS选择走法
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
        elif mcts_or_net == "net":
            # 直接用网络选择走法
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)
            win_rate = value[0, 0]  # 网络预测的价值
            # 黑方回合则翻转策略
            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            # 获取合法走法
            moves = GameBoard.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            # 计算合法走法的概率（归一化）
            tot_p = 1e-8
            action_probs = action_probs.flatten()
            act_prob_dict = defaultdict(float)
            for action in moves:
                mov_p = action_probs[label2i[action]]
                act_prob_dict[action] = mov_p
                tot_p += mov_p
            # 归一化
            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p
            # 选择概率最大的走法
            action = max(act_prob_dict.items(), key=lambda node: node[1])[0]

        # 打印赢率信息
        print(f'当前玩家 {self.game_borad.current_player} 的赢率: {win_rate:.4f}')
        # 更新棋盘状态
        last_state = self.game_borad.state
        print(f"{self.game_borad.current_player} 走了: {action} [第 {self.game_borad.round} 步]")
        self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        # 切换玩家
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        # 更新无吃子回合数
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        # 打印棋盘
        self.game_borad.print_borad(self.game_borad.state)

        # 转换走法为坐标差（用于界面显示）
        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))
        src = action[0:2]
        dst = action[2:4]
        src_x = int(x_trans[src[0]])
        src_y = int(src[1])
        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        return (src_x, src_y, dst_x - src_x, dst_y - src_y), win_rate

    def selfplay(self):
        """自我对弈（生成训练数据）"""
        self.game_borad.reload()  # 重置棋盘
        states, mcts_probs, current_players = [], [], []  # 存储状态、策略、当前玩家
        z = None  # 胜负标记
        game_over = False
        winnner = ""
        start_time = time.time()

        while not game_over:
            # 获取当前玩家的走法
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            # 翻转棋盘（统一用红方视角存储）
            state, palyer = self.mcts.try_flip(self.game_borad.state, self.game_borad.current_player,
                                              self.mcts.is_black_turn(self.game_borad.current_player))
            states.append(state)  # 记录状态

            # 处理策略概率（根据当前玩家是否黑方进行翻转）
            prob = np.zeros(labels_len)
            if self.mcts.is_black_turn(self.game_borad.current_player):
                for idx in range(len(probs[0][0])):
                    # 翻转走法标签
                    act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    prob[label2i[act]] = probs[0][1][idx]
            else:
                for idx in range(len(probs[0][0])):
                    prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
            mcts_probs.append(prob)  # 记录策略
            current_players.append(self.game_borad.current_player)  # 记录当前玩家

            # 更新棋盘状态
            last_state = self.game_borad.state
            self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
            self.game_borad.round += 1
            # 切换玩家
            self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
            # 更新无吃子回合数
            if is_kill_move(last_state, self.game_borad.state) == 0:
                self.game_borad.restrict_round += 1
            else:
                self.game_borad.restrict_round = 0

            # 检查游戏是否结束
            if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
                z = np.zeros(len(current_players))
                if self.game_borad.state.find('K') == -1:
                    winnner = "b"  # 黑方赢
                if self.game_borad.state.find('k') == -1:
                    winnner = "w"  # 红方赢
                # 标记胜负（胜者为1，败者为-1）
                z[np.array(current_players) == winnner] = 1.0
                z[np.array(current_players) != winnner] = -1.0
                game_over = True
                print(f"游戏结束。赢家是 {winnner}，共 {self.game_borad.round - 1} 步")
            elif self.game_borad.restrict_round >= 60:
                z = np.zeros(len(current_players))  # 平局，价值为0
                game_over = True
                print(f"游戏结束。平局，共 {self.game_borad.round - 1} 步")
            if game_over:
                self.mcts.reload()  # 重置MCTS树

        print(f"用时 {time.time() - start_time} 秒")
        return zip(states, mcts_probs, z), len(z)