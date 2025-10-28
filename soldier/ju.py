import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class ju(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RRS.gif"
            else:
                return "images/jured.gif"
        else:
            if self.is_red:
                return "images/RR.gif"
            else:
                return "images/ju.gif"

    def can_move(self, board, dx, dy):
        if dx != 0 and dy != 0:
            # 不是直线。。。
            return False
        nx, ny = self.x + dx, self.y + dy
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False
        if (nx, ny) in board.pieces:
            if board.pieces[nx, ny].is_red == self.is_red:
                # 卡住了
                return False
        cnt = self.count_pieces(board, self.x, self.y, dx, dy)
        if (nx, ny) not in board.pieces:
            if cnt!= 0:
                # 卡住了
                return False
        else:
            if cnt != 0:
                # 不是直线啊哥们
                return False
            print ('kill a chessman')
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

