import sys

from battlefield.displaychess import displaychess

sys.path.append("...")

class bing(displaychess):

    def image_name(self):
        if self.selected:
            if self.is_red:
                return "images/RPS.gif"
            else:
                return "images/zured.gif"
        else:
            if self.is_red:
                return "images/RP.gif"
            else:
                return "images/zu.gif"

    def can_move(self, board, dx, dy):
        nx, ny = self.x + dx, self.y + dy
        x, y = self.x, self.y

        # 1. 必须走一格
        if abs(dx) + abs(dy) != 1:
            return False

        # 2. 禁止后退（根据实际y轴方向，假设红方初始y值小，向上y增大）
        if self.is_red:
            if dy < 0:  # 红方后退（向下，y减小）
                return False
        else:
            if dy > 0:  # 黑方后退（向上，y增大）
                return False

        # 3. 未过河禁横向移动（若红方初始y=0~4，未过河y<5）
        if dx != 0:
            if (self.is_red and y < 5) or (not self.is_red and y > 4):
                return False

        # 4. 边界检查
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False

        # 5. 己方棋子检查
        if (nx, ny) in board.pieces and board.pieces[(nx, ny)].is_red == self.is_red:
            return False

        return True

    def get_move_locs(self, board):
        """返回所有合法移动位置的坐标列表"""
        move_locs = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        for dx, dy in directions:
            if self.can_move(board, dx, dy):
                nx = self.x + dx
                ny = self.y + dy
                move_locs.append((nx, ny))
        return move_locs

    def __init__(self, x, y, is_red, direction):
        displaychess.__init__(self, x, y, is_red, direction)

    def display(self):
        sys.stdout.write('B')
