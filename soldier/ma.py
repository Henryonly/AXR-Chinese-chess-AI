import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class ma(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RNS.gif"
            else:
                return "images/mared.gif"
        else:
            if self.is_red:
                return "images/RN.gif"
            else:
                return "images/ma.gif"

    def can_move(self, board, dx, dy):
        x, y = self.x, self.y
        nx, ny = x+dx, y+dy
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False
        if dx == 0 or dy == 0:
            # 不走直线。。。
            return False
        if abs(dx) + abs(dy) !=3:
            # 不是走的日啊
            return False
        if (nx, ny) in board.pieces:
            if board.pieces[nx, ny].is_red == self.is_red:
                # 被自己卡住力
                return False
        if (x if abs(dx) ==1 else x+dx/2, y if abs(dy) ==1 else y+ (dy/2)) in board.pieces:
            # 卡住力
            return False
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

