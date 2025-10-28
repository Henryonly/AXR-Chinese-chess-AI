import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class shi(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RAS.gif"
            else:
                return "images/shired.gif"
        else:
            if self.is_red:
                return "images/RA.gif"
            else:
                return "images/shi.gif"

    def can_move(self, board, dx, dy):
        nx, ny = self.x + dx, self.y + dy  # 目标位置坐标

        # 1. 首先判断是否斜走一步（必须 dx=±1 且 dy=±1）
        if abs(dx) != 1 or abs(dy) != 1:
            return False  # 不是斜走一步，非法

        # 2. 判断目标位置是否在己方九宫格内
        if self.is_red:  # 红方（假设 is_north() 等价于 is_red）
            # 红方九宫：x在3-5之间，y在0-2之间
            if not (3 <= nx <= 5 and 0 <= ny <= 2):
                return False
        else:  # 黑方（is_south()）
            # 黑方九宫：x在3-5之间，y在7-9之间
            if not (3 <= nx <= 5 and 7 <= ny <= 9):
                return False

        # 3. 判断目标位置是否有己方棋子
        if (nx, ny) in board.pieces:
            if board.pieces[(nx, ny)].is_red == self.is_red:
                return False  # 目标位置是己方棋子，非法

        # 所有条件满足，允许移动
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

