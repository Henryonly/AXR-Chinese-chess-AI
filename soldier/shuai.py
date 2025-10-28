import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class shuai(displaychess):

    is_king = True
    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RKS.gif"
            else:
                return "images/jiangred.gif"
        else:
            if self.is_red:
                return "images/RK.gif"
            else:
                return "images/jiang.gif"

    def can_move(self, board, dx, dy):
        nx, ny = self.x + dx, self.y + dy  # 目标位置

        # 1. 先判断是否是“将帅照面”（特殊规则）
        if self.is_king:  # 帅将的特殊走法
            # 检查是否在同一列（x相同）
            if nx == self.x:  # dx必须为0（沿y轴移动）
                # 目标位置是对方帅将
                if (nx, ny) in board.pieces and board.pieces[(nx, ny)].is_king:
                    # 判断中间是否有棋子阻挡（实现count_pieces的功能）
                    min_y = min(self.y, ny)
                    max_y = max(self.y, ny)
                    has_block = False
                    for y in range(min_y + 1, max_y):
                        if (nx, y) in board.pieces:
                            has_block = True
                            break
                    if not has_block:
                        return True  # 中间无棋子，可飞将

        # 2. 常规移动：判断是否在己方九宫格内
        if self.is_red:  # 红方帅
            if not (3 <= nx <= 5 and 0 <= ny <= 2):
                return False  # 超出九宫
        else:  # 黑方将
            if not (3 <= nx <= 5 and 7 <= ny <= 9):
                return False  # 超出九宫

        # 3. 常规移动：只能走一步（上下左右）
        if abs(dx) + abs(dy) != 1:
            return False  # 不是一步移动

        # 4. 目标位置不能有己方棋子
        if (nx, ny) in board.pieces:
            if board.pieces[(nx, ny)].is_red == self.is_red:
                return False  # 目标是己方棋子

        # 所有条件满足
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

