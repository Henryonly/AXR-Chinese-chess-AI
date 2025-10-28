import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class xiang(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RBS.gif"
            else:
                return "images/xiangred.gif"
        else:
            if self.is_red:
                return "images/RB.gif"
            else:
                return "images/xiang.gif"

    def can_move(self, board, dx, dy):
        x, y = self.x, self.y
        nx, ny = x + dx, y + dy

        # 1. 检查是否超出棋盘范围
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False

        # 2. 检查目标位置是否有己方棋子
        if (nx, ny) in board.pieces:
            if board.pieces[(nx, ny)].is_red == self.is_red:
                return False  # 目标是己方棋子，不能移动

        # 3. 关键：禁止过河（红方 ny ≤ 4，黑方 ny ≥ 5）
        if self.is_red:  # 红方象
            if ny > 4:  # 红方 ny > 4 即为过河
                return False
        else:  # 黑方相
            if ny < 5:  # 黑方 ny < 5 即为过河
                return False

        # 4. 检查是否走“田”字（dx=±2 且 dy=±2）
        if abs(dx) != 2 or abs(dy) != 2:
            return False  # 不是田字步

        # 5. 检查“田”字中心是否有棋子（绊象腿）
        # 计算田字中心坐标（dx/2 和 dy/2 取整数）
        center_x = x + (dx // 2)
        center_y = y + (dy // 2)
        if (center_x, center_y) in board.pieces:
            return False  # 中心有棋子，绊腿了

        # 所有条件满足，允许移动
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

