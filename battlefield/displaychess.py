class displaychess:

    # 一些重要指标
    selected=False
    king=False

    # 位置、颜色、方向
    def __init__(self,x,y,is_red,dir):
        self.x=x
        self.y=y
        self.is_red=is_red
        self.dir=dir
    #方向是啥啊
    def is_north(self):
        return self.dir=='N'
    def is_south(self):
        return self.dir=='S'

    #下面是移动相关 board是对应的坐标数组

    def move_whether(self,board):
        moves=[] #移动的记忆库
        for x in range(9):
            for y in range(10):
                if (x,y) in board.pieces and board.pieces[x,y].is_red==self.is_red:
                    #最起码要在棋盘里面而且选中了。。。
                    continue
                if self.can_move(board,x-self.x,y-self.y):
                    moves.append((x,y))

    def move(self,board,dx,dy):
        nx=dx+self.x
        ny=dy+self.y
        if (nx,ny) in board.pieces: # 移动别移动到天上去了
            board.remove(nx,ny)
        board.remove(self.x,self.y)
        # 也就是这个棋子它动了！！！去(nx,ny)了！！！
        self.x=nx
        self.y=ny
        board.pieces[self.x,self.y]=self # 别忘了更新棋盘
        return True

    def count_pieces(self,board,x,y,dx,dy):
    # 用来看得走多远才能不在同一条线上
        if dx==0:
            sx=0
        else:
            sx=dx/abs(dx)
        if dy==0:
            sy=0
        else:
            sy=dy/abs(dy)

        nx=x+dx
        ny=y+dy
        x=x+sx
        y=y+sy
        count=0
        while x!=nx or y!=ny:
            # 最多三个循环，而且这个是可以走过本来要的地方的，所以循环次数不会超过走的最大横向/纵向距离
            if (x,y) in board.pieces:
                count+=1
            x+=sx
            y+=sy
        return count