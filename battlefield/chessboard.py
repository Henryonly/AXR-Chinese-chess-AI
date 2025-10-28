import sys
from pathlib import Path

# 定位项目根目录（RL_chess文件夹）
# 这里通过当前文件路径向上两级找到根目录，方便后续导入模块
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
# 导入所有棋子类（兵、帅、炮、士、相、马、车）
from soldier.bing import *
from soldier.shuai import *
from soldier.pao import *
from soldier.shi import *
from soldier.xiang import *
from soldier.ma import *
from soldier.ju import *

class chessboard:
    # 用字典存棋盘上的棋子，key是(x,y)坐标，value是棋子实例
    pieces = dict()
    # 记录当前选中的棋子，没选中时为None
    selected_piece = None

    def __init__(self, north_is_red = True):
        # 初始化棋盘，摆放红黑双方的棋子
        # north_is_red参数控制上方是不是红方

        # 上方（north）棋子摆放
        # 帅
        chessboard.pieces[4, 0] = shuai(4, 0, north_is_red, "north")
        # 兵
        chessboard.pieces[0, 3] = bing(0, 3, north_is_red, "north")
        chessboard.pieces[2, 3] = bing(2, 3, north_is_red, "north")
        chessboard.pieces[4, 3] = bing(4, 3, north_is_red, "north")
        chessboard.pieces[6, 3] = bing(6, 3, north_is_red, "north")
        chessboard.pieces[8, 3] = bing(8, 3, north_is_red, "north")
        # 炮
        chessboard.pieces[1, 2] = pao(1, 2, north_is_red, "north")
        chessboard.pieces[7, 2] = pao(7, 2, north_is_red, "north")
        # 士
        chessboard.pieces[3, 0] = shi(3, 0, north_is_red, "north")
        chessboard.pieces[5, 0] = shi(5, 0, north_is_red, "north")
        # 相
        chessboard.pieces[2, 0] = xiang(2, 0, north_is_red, "north")
        chessboard.pieces[6, 0] = xiang(6, 0, north_is_red, "north")
        # 马
        chessboard.pieces[1, 0] = ma(1, 0, north_is_red, "north")
        chessboard.pieces[7, 0] = ma(7, 0, north_is_red, "north")
        # 车
        chessboard.pieces[0, 0] = ju(0, 0, north_is_red, "north")
        chessboard.pieces[8, 0] = ju(8, 0, north_is_red, "north")

        # 下方（south）棋子摆放，颜色和上方相反
        # 帅
        chessboard.pieces[4, 9] = shuai(4, 9, not north_is_red, "south")
        # 兵
        chessboard.pieces[0, 6] = bing(0, 6, not north_is_red, "south")
        chessboard.pieces[2, 6] = bing(2, 6, not north_is_red, "south")
        chessboard.pieces[4, 6] = bing(4, 6, not north_is_red, "south")
        chessboard.pieces[6, 6] = bing(6, 6, not north_is_red, "south")
        chessboard.pieces[8, 6] = bing(8, 6, not north_is_red, "south")
        # 炮
        chessboard.pieces[1, 7] = pao(1, 7, not north_is_red, "south")
        chessboard.pieces[7, 7] = pao(7, 7, not north_is_red, "south")
        # 士
        chessboard.pieces[3, 9] = shi(3, 9, not north_is_red, "south")
        chessboard.pieces[5, 9] = shi(5, 9, not north_is_red, "south")
        # 相
        chessboard.pieces[2, 9] = xiang(2, 9, not north_is_red, "south")
        chessboard.pieces[6, 9] = xiang(6, 9, not north_is_red, "south")
        # 马
        chessboard.pieces[1, 9] = ma(1, 9, not north_is_red, "south")
        chessboard.pieces[7, 9] = ma(7, 9, not north_is_red, "south")
        # 车
        chessboard.pieces[0, 9] = ju(0, 9, not north_is_red, "south")
        chessboard.pieces[8, 9] = ju(8, 9, not north_is_red, "south")

    def can_move(self, x, y, dx, dy):
        # 检查(x,y)位置的棋子能不能移动dx、dy的距离
        return self.pieces[x, y].can_move(self, dx, dy)

    def move(self, x, y, dx, dy):
        # 移动(x,y)位置的棋子，移动距离为dx、dy
        return self.pieces[x, y].move(self, dx, dy)

    def remove(self, x, y):
        # 从棋盘上删掉(x,y)位置的棋子
        del self.pieces[x, y]

    def select(self, x, y, player_is_red):
        # 处理选子和移动的逻辑
        # player_is_red表示当前走棋的是不是红方

        # 还没选中棋子时，点自己的棋子就选中它
        if not self.selected_piece:
            if (x, y) in self.pieces and self.pieces[x, y].is_red == player_is_red:
                self.pieces[x, y].selected = True
                self.selected_piece = self.pieces[x, y]
            return False, None

        # 已经选中棋子，点了空白处——尝试移动
        if not (x, y) in self.pieces:
            if self.selected_piece:
                ox, oy = self.selected_piece.x, self.selected_piece.y
                if self.can_move(ox, oy, x-ox, y-oy):
                    self.move(ox, oy, x-ox, y-oy)
                    self.pieces[x,y].selected = False
                    self.selected_piece = None
                    return True, (ox, oy, x, y)
            return False, None

        # 点了已经选中的棋子，不做操作
        if self.pieces[x, y].selected:
            return False, None

        # 点了对方的棋子——尝试吃子
        if self.pieces[x, y].is_red != player_is_red:
            ox, oy = self.selected_piece.x, self.selected_piece.y
            if self.can_move(ox, oy, x-ox, y-oy):
                self.move(ox, oy, x-ox, y-oy)
                self.pieces[x,y].selected = False
                self.selected_piece = None
                return True, (ox, oy, x, y)
            return False, None

        # 点了自己的另一个棋子——切换选中
        for key in self.pieces.keys():
            self.pieces[key].selected = False
        self.pieces[x, y].selected = True
        self.selected_piece = self.pieces[x,y]
        return False, None