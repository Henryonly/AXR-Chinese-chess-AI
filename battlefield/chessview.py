import tkinter
import time


def board_coord(x):
    # 把棋盘格子坐标转换成界面上的像素坐标（方便画图）
    return 30 + 40 * x


class chessview:
    # 初始化主窗口
    root = tkinter.Tk()
    root.title("Chinese Chess")  # 窗口标题设为"中国象棋"
    root.resizable(0, 0)  # 窗口大小固定，不能拉伸
    # 创建画布，宽373像素，高410像素，用来画棋盘和棋子
    can = tkinter.Canvas(root, width=373, height=410)
    can.pack(expand=tkinter.YES, fill=tkinter.BOTH)  # 画布填满窗口
    # 加载棋盘背景图（images文件夹下的qipan.gif）
    img = tkinter.PhotoImage(file="images/qipan.gif")
    can.create_image(0, 0, image=img, anchor=tkinter.NW)  # 左上角对齐显示背景图
    piece_images = dict()  # 存棋子图片，key是(x,y)坐标，value是图片对象
    move_images = []  # 存可移动位置的标记图片（比如那个圈圈）

    def draw_board(self, board):
        # 刷新棋盘显示：先清空之前的棋子和标记
        self.piece_images.clear()
        self.move_images = []
        pieces = board.pieces  # 获取棋盘上所有棋子

        # 遍历所有棋子，逐个画到画布上
        for (x, y) in pieces.keys():
            # 根据棋子类型加载对应的图片
            self.piece_images[x, y] = tkinter.PhotoImage(file=pieces[x, y].image_name())
            # 把棋子画在对应的像素位置
            self.can.create_image(board_coord(x), board_coord(y), image=self.piece_images[x, y])

        # 如果有选中的棋子，画出它能走的位置
        if board.selected_piece:
            # 遍历所有可移动的位置
            for (x, y) in board.selected_piece.can_move(board):
                # 加载可移动标记的图片（OOS.gif）
                self.move_images.append(tkinter.PhotoImage(file="images/OOS.gif"))
                # 画到对应位置
                self.can.create_image(board_coord(x), board_coord(y), image=self.move_images[-1])

    def disp_hint_on_board(self, action, percentage):
        # 在棋盘上显示提示：高亮源位置、目标位置，以及概率值
        board = self.board
        # 先取消所有棋子的选中状态
        for key in board.pieces.keys():
            board.pieces[key].selected = False
        board.selected_piece = None  # 清空选中的棋子

        # 重新画一下棋盘背景（相当于清屏）
        self.can.create_image(0, 0, image=self.img, anchor=tkinter.NW)
        self.draw_board(board)  # 重新画棋子

        # 把字母坐标（a-i）转换成数字（0-8）
        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
        # 解析动作：前两位是源位置，后两位是目标位置（比如"a1b2"表示从a1到b2）
        src = action[0:2]
        dst = action[2:4]

        # 转成数字坐标
        src_x = int(x_trans[src[0]])
        src_y = int(src[1])
        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        pieces = board.pieces
        # 高亮显示源位置的棋子
        if (src_x, src_y) in pieces.keys():
            # 加载选中状态的棋子图片
            self.piece_images[src_x, src_y] = tkinter.PhotoImage(file=pieces[src_x, src_y].get_selected_image())
            # 画到源位置
            self.can.create_image(board_coord(src_x), board_coord(src_y), image=self.piece_images[src_x, src_y])

        # 处理目标位置：如果有棋子就高亮，否则画标记，同时显示概率
        if (dst_x, dst_y) in pieces.keys():
            # 目标位置有棋子（可能是吃子），高亮它
            self.piece_images[dst_x, dst_y] = tkinter.PhotoImage(file=pieces[dst_x, dst_y].get_selected_image())
            self.can.create_image(board_coord(dst_x), board_coord(dst_y), image=self.piece_images[dst_x, dst_y])
            # 在目标位置显示概率（保留3位小数）
            self.can.create_text(board_coord(dst_x), board_coord(dst_y), text="{:.3f}".format(percentage))
            # 记录一下文字位置，方便后续清除
            self.last_text_x = dst_x
            self.last_text_y = dst_y
        else:
            # 目标位置是空的，画可移动标记
            self.move_images.append(tkinter.PhotoImage(file="images/OOS.gif"))
            self.can.create_image(board_coord(dst_x), board_coord(dst_y), image=self.move_images[-1])
            # 显示概率
            self.can.create_text(board_coord(dst_x), board_coord(dst_y), text="{:.3f}".format(percentage))
            self.last_text_x = dst_x
            self.last_text_y = dst_y
            self.print_text_flag = True  # 标记有文字需要清除

    def print_all_hint(self, sorted_move_probs):
        # 在列表框里显示所有推荐走法和概率
        self.lb.delete(0, "end")  # 先清空列表
        # 逐个添加推荐项
        for item in sorted_move_probs:
            self.lb.insert("end", item)
        self.lb.pack()  # 显示列表

    def showMsg(self, msg):
        # 在控制台打印消息，同时更新窗口标题
        print(msg)
        self.root.title(msg)

    def printList(self, event):
        # 处理列表框的选择事件：点击某个推荐走法，就在棋盘上显示提示
        w = event.widget  # 获取触发事件的控件（这里是列表框）
        index = int(w.curselection()[0])  # 获取选中项的索引
        value = w.get(index)  # 获取选中项的内容
        print(value)  # 控制台打印一下
        # 在棋盘上显示这个走法的提示
        self.disp_hint_on_board(value[0], value[1])

    def __init__(self, control, board):
        # 初始化界面，关联游戏控制逻辑和棋盘
        self.control = control
        # 不是AIvsAI模式时，绑定鼠标点击事件（人走棋用）
        if self.control.game_mode != 2:
            self.can.bind('<Button-1>', self.control.callback)

        # 创建列表框（显示推荐走法）和滚动条
        self.lb = tkinter.Listbox(chessview.root, selectmode="browse")  # 单选模式
        self.scr1 = tkinter.Scrollbar(chessview.root)
        # 关联列表框和滚动条
        self.lb.configure(yscrollcommand=self.scr1.set)
        self.scr1['command'] = self.lb.yview
        # 滚动条放右边，占满高度
        self.scr1.pack(side='right', fill="y")
        # 列表框水平方向填满
        self.lb.pack(fill="x")

        # 绑定列表框选择事件
        self.lb.bind('<<ListboxSelect>>', self.printList)
        self.board = board  # 保存棋盘对象
        # 记录上次显示文字的位置，初始为(0,0)
        self.last_text_x = 0
        self.last_text_y = 0
        self.print_text_flag = False  # 是否有文字需要清除的标记

    def start(self):
        # 启动界面循环
        if self.control.game_mode == 2:
            # AIvsAI模式：循环刷新界面，按延迟时间更新
            self.root.update()  # 先刷新一次
            time.sleep(self.control.delay)  # 等一下
            while True:
                # 调用游戏控制逻辑，判断是否结束
                game_end = self.control.game_mode_2()
                self.root.update()  # 刷新界面
                time.sleep(self.control.delay)  # 按设定的延迟等待
                if game_end:
                    # 游戏结束，等一会儿再退出
                    time.sleep(self.control.end_delay)
                    self.quit()
                    return
        else:
            # 人参与的模式：启动tkinter主循环
            tkinter.mainloop()

    def quit(self):
        # 退出游戏，关闭窗口
        self.root.quit()