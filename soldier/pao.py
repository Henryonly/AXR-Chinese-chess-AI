import sys
from battlefield.displaychess import displaychess
sys.path.append("...")

class pao(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RCS.gif"
            else:
                return "images/paored.gif"
        else:
            if self.is_red:
                return "images/RC.gif"
            else:
                return "images/pao.gif"

    def can_move(self, board, dx, dy):
        if dx != 0 and dy != 0:
            # 只能走直线啊
            return False
        nx, ny = self.x + dx, self.y + dy
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False
        if (nx, ny) in board.pieces:
            if board.pieces[nx, ny].is_red == self.is_red:
                # 卡住了动不了
                return False
        cnt = self.count_pieces(board, self.x, self.y, dx, dy)
        # 中间隔多少个子，看一看~~~
        if (nx, ny) not in board.pieces:
            if cnt!= 0:
                # 卡住了
                return False
        else:
            if cnt != 1:
                # 杀不掉
                return False
        return True

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

