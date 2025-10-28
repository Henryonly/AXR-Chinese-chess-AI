import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 禁用TensorFlow 2.x特性，用1.x的方式运行
import numpy as np
import os

# 定义参数服务器（Parameter Server）相关的操作类型
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


class policy_value_network(object):
    def __init__(self, num_gpus=1, res_block_nums=7):
        # 初始化模型参数
        self.num_gpus = num_gpus  # 使用的GPU数量
        self.save_dir = "./gpu_models"  # 模型保存路径
        self.is_logging = True  # 是否开启日志记录
        self.res_block_nums = res_block_nums  # 残差块的数量

        # 重置默认计算图，确保每次初始化都是新图
        tf.reset_default_graph()

        # 配置TensorFlow会话（Session）参数
        config = tf.ConfigProto(
            inter_op_parallelism_threads=4,  # 控制不同操作间的并行线程数
            intra_op_parallelism_threads=4)  # 控制同一操作内的并行线程数
        config.gpu_options.allow_growth = True  # 按需分配GPU内存，不一次性占满
        config.allow_soft_placement = True  # 自动选择可用设备（比如GPU满了就用CPU）
        self.sess = tf.Session(config=config)  # 创建会话

        # 所有变量和操作先放在CPU上统一管理
        with tf.device('/cpu:0'):
            # 网络基本配置
            self.filters_size = 128  # 卷积核数量（也可以设为256）
            self.prob_size = 2086  # 策略头输出维度（所有可能走法的数量）
            self.digest = None  # 预留字段，暂未使用

            # 定义输入占位符
            self.training = tf.placeholder(tf.bool, name='training')  # 标记是否训练模式
            # 输入棋盘状态：[批次大小, 9列, 10行, 14通道]（14通道表示不同棋子和状态）
            self.inputs_ = tf.placeholder(tf.float32, [None, 9, 10, 14], name='inputs')
            self.c_l2 = 0.0001  # L2正则化系数
            self.momentum = 0.9  # 动量优化器的动量参数
            self.global_norm = 100  # 梯度裁剪的最大范数
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # 学习率（动态调整）

            # 全局步数（记录训练迭代次数）
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            tf.summary.scalar('learning_rate', self.learning_rate)  # 记录学习率到日志

            # 定义优化器（用动量优化器，带Nesterov加速）
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)

            # 定义标签占位符（策略目标和价值目标）
            self.pi_ = tf.placeholder(tf.float32, [None, self.prob_size], name='pi')  # 策略目标（走法概率）
            self.z_ = tf.placeholder(tf.float32, [None, 1], name='z')  # 价值目标（胜负结果）

            # 将输入数据按GPU数量拆分（多GPU并行训练用）
            inputs_batches = tf.split(self.inputs_, self.num_gpus, axis=0)
            pi_batches = tf.split(self.pi_, self.num_gpus, axis=0)
            z_batches = tf.split(self.z_, self.num_gpus, axis=0)

            tower_grads = [None] * self.num_gpus  # 存储每个GPU计算的梯度

            # 初始化损失、准确率和输出头
            self.loss = 0
            self.accuracy = 0
            self.policy_head = []  # 策略头输出（每个GPU一个）
            self.value_head = []  # 价值头输出（每个GPU一个）

            # 多GPU训练：每个GPU负责一部分数据
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    # 指定当前操作在第i个GPU上运行，参数存在CPU上
                    with tf.device(self.assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                        with tf.name_scope('TOWER_{}'.format(i)) as scope:  # 命名空间，方便日志查看
                            # 获取当前GPU对应的批次数据
                            inputs_batch, pi_batch, z_batch = inputs_batches[i], pi_batches[i], z_batches[i]
                            # 计算当前GPU的损失
                            loss = self.tower_loss(inputs_batch, pi_batch, z_batch, i)
                            tf.get_variable_scope().reuse_variables()  # 复用变量（多GPU共享参数）
                            # 计算当前GPU的梯度
                            grad = optimizer.compute_gradients(loss)
                            tower_grads[i] = grad

            # 平均所有GPU的损失和准确率
            self.loss /= self.num_gpus
            self.accuracy /= self.num_gpus
            # 平均所有GPU的梯度（多GPU训练核心）
            grads = self.average_gradients(tower_grads)
            # 梯度裁剪：防止梯度爆炸
            clipped_grads, self.norm = tf.clip_by_global_norm(
                [g for g, _ in grads], self.global_norm)

            # 检查梯度是否有NaN（数值稳定性保障）
            grad_check = [tf.check_numerics(g, message='NaN Found!') for g in clipped_grads]
            with tf.control_dependencies(grad_check):
                # 应用梯度更新参数
                self.train_op = optimizer.apply_gradients(
                    zip(clipped_grads, [v for _, v in grads]),
                    global_step=self.global_step, name='train_step')

            # 日志相关：记录变量分布、梯度分布等
            if self.is_logging:
                for grad, var in grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)

            # 合并所有日志操作
            self.summaries_op = tf.summary.merge_all()
            # 创建日志写入器（训练和测试日志分开）
            self.train_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), "cchesslogs/train"), self.sess.graph)
            self.test_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), "cchesslogs/test"), self.sess.graph)

            # 初始化所有变量，并尝试加载已保存的模型
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()  # 用于保存/加载模型
            self.train_restore()  # 尝试加载历史模型

    def tower_loss(self, inputs_batch, pi_batch, z_batch, i):
        # 单个GPU上的损失计算（构建网络并计算损失）
        with tf.variable_scope('init'):
            # 初始卷积层：3x3卷积核，128个输出通道， SAME填充（保持尺寸）
            print("初始输入形状:", inputs_batch.shape)
            layer = tf.keras.layers.Conv2D(filters=self.filters_size, kernel_size=3, padding='SAME')(inputs_batch)
            print("初始卷积后形状:", layer.shape)
            # 批归一化（不使用偏移量，数值稳定性）
            layer = tf.keras.layers.BatchNormalization(center=False, epsilon=1e-5)(layer)

        # 堆叠残差块（数量由res_block_nums指定）
        with tf.variable_scope("residual_block", reuse=tf.AUTO_REUSE):
            for _ in range(self.res_block_nums):
                layer = self.residual_block(layer)

        # 策略头：输出走法概率分布
        with tf.variable_scope("policy_head"):
            # 1x1卷积压缩通道数到2
            policy_head = tf.keras.layers.Conv2D(filters=2, kernel_size=1, padding='SAME')(layer)
            policy_head = tf.keras.layers.BatchNormalization(center=False, epsilon=1e-5, axis=-1)(policy_head)
            policy_head = tf.keras.layers.ReLU()(policy_head)  # 激活函数

            # 展平成一维向量，再通过全连接层输出策略概率（2086种走法）
            policy_head = tf.reshape(policy_head, [-1, 9 * 10 * 2])
            policy_head = tf.keras.layers.Dense(units=self.prob_size, activation=None)(policy_head)
            self.policy_head.append(policy_head)  # 保存当前GPU的策略输出

        # 价值头：输出当前局面的胜负估计（-1到1之间）
        with tf.variable_scope("value_head"):
            # 1x1卷积压缩通道数到1
            value_head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding='SAME')(layer)
            value_head = tf.keras.layers.BatchNormalization(center=False, epsilon=1e-5, axis=-1)(value_head)
            value_head = tf.keras.layers.ReLU()(value_head)  # 激活函数

            # 展平后通过全连接层，最后用tanh输出（范围-1到1）
            value_head = tf.reshape(value_head, [-1, 9 * 10 * 1])
            value_head = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)(value_head)
            value_head = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)(value_head)
            self.value_head.append(value_head)  # 保存当前GPU的价值输出

        # 计算损失
        with tf.variable_scope("loss"):
            # 策略损失：交叉熵（衡量预测概率与目标概率的差距）
            policy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi_batch, logits=policy_head)
            policy_loss = tf.reduce_mean(policy_loss)  # 平均到批次

            # 价值损失：均方误差（衡量预测价值与实际胜负的差距）
            value_loss = tf.losses.mean_squared_error(labels=z_batch, predictions=value_head)
            value_loss = tf.reduce_mean(value_loss)  # 平均到批次
            tf.summary.scalar('mse_tower_{}'.format(i), value_loss)  # 记录MSE到日志

            # L2正则化损失（防止过拟合）
            regular_variables = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(var) * self.c_l2 for var in regular_variables])

            # 总损失 = 价值损失 + 策略损失 + 正则化损失
            loss = value_loss + policy_loss + l2_loss
            self.loss += loss  # 累加到总损失（后续会除以GPU数量取平均）
            tf.summary.scalar('loss_tower_{}'.format(i), loss)  # 记录总损失到日志

        # 计算准确率（策略预测的准确率）
        with tf.variable_scope("accuracy"):
            # 比较预测的最佳走法与目标走法是否一致
            correct_prediction = tf.equal(tf.argmax(policy_head, 1), tf.argmax(pi_batch, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)  # 转成数值（0或1）
            accuracy = tf.reduce_mean(correct_prediction, name='accuracy')  # 平均准确率
            self.accuracy += accuracy  # 累加到总准确率
            tf.summary.scalar('move_accuracy_tower_{}'.format(i), accuracy)  # 记录准确率到日志
        return loss

    def assign_to_device(self, device, ps_device='/cpu:0'):
        # 设备分配函数：参数存在ps_device（CPU），计算操作在device（GPU）
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            # 如果是参数相关操作，放到ps_device（CPU）
            if node_def.op in PS_OPS:
                return "/" + ps_device
            else:
                return device  # 其他操作放到指定设备（GPU）

        return _assign

    def average_gradients(self, tower_grads):
        # 平均多个GPU的梯度（多GPU训练时用）
        average_grads = []
        # 按变量分组，每个变量收集所有GPU的梯度
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, var in grad_and_vars:
                # 给每个梯度增加一个维度，方便后续拼接
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            # 拼接所有GPU的梯度，再求平均
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # 保留变量信息，返回（平均梯度，变量）
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def residual_block(self, in_layer):
        # 残差块：包含两个卷积层 + 跳跃连接（解决深层网络梯度消失问题）
        print("残差块输入形状:", in_layer.shape)
        orig = tf.identity(in_layer)  # 保存原始输入，用于跳跃连接

        # 第一次卷积 + 批归一化 + 激活
        with tf.variable_scope('conv1'):
            # 3x3卷积核，输入通道数=当前层通道数，输出通道数=128
            kernel = tf.get_variable(
                'kernel',
                shape=[3, 3, in_layer.shape[-1].value, self.filters_size],
                initializer=tf.glorot_uniform_initializer()  # 初始化权重
            )
            layer = tf.nn.conv2d(
                input=in_layer,
                filter=kernel,
                strides=[1, 1, 1, 1],  # 步长1，不改变尺寸
                padding='SAME'  # 保持输出尺寸与输入一致
            )

        # 批归一化（用TF1.x原生接口，不使用偏移量）
        with tf.variable_scope('bn1'):
            mean, var = tf.nn.moments(layer, axes=[0, 1, 2])  # 计算批次的均值和方差
            layer = tf.nn.batch_normalization(
                x=layer,
                mean=mean,
                variance=var,
                offset=None,  # center=False，不需要偏移
                scale=None,  # 暂不使用缩放
                variance_epsilon=1e-5  # 防止方差为0
            )
        layer = tf.nn.relu(layer)  # ReLU激活
        print("第一次卷积后形状:", layer.shape)

        # 第二次卷积 + 批归一化
        with tf.variable_scope('conv2'):
            kernel = tf.get_variable(
                'kernel',
                shape=[3, 3, self.filters_size, self.filters_size],  # 输入输出都是128通道
                initializer=tf.glorot_uniform_initializer()
            )
            layer = tf.nn.conv2d(
                input=layer,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

        # 第二次批归一化
        with tf.variable_scope('bn2'):
            mean, var = tf.nn.moments(layer, axes=[0, 1, 2])
            layer = tf.nn.batch_normalization(
                x=layer,
                mean=mean,
                variance=var,
                offset=None,
                scale=None,
                variance_epsilon=1e-5
            )

        # 跳跃连接：原始输入 + 卷积后的输出，再激活
        out = tf.keras.layers.Add()([orig, layer])  # 元素相加
        out = tf.keras.layers.ReLU()(out)  # ReLU激活
        print("残差块输出形状:", out.shape)
        return out

    def train_restore(self):
        # 尝试加载已保存的模型（用于继续训练）
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)  # 目录不存在则创建
        checkpoint = tf.train.get_checkpoint_state(self.save_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # 加载最新的模型
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            print("成功加载模型:", tf.train.latest_checkpoint(self.save_dir))
        else:
            print("未找到历史模型，将从头开始训练")

    def restore(self, file):
        # 加载指定路径的模型（用于评估或测试）
        print("从{}加载模型".format(file))
        self.saver.restore(self.sess, file)

    def save(self, in_global_step):
        # 保存模型（带当前训练步数）
        save_path = self.saver.save(
            self.sess,
            os.path.join(self.save_dir, 'best_model.ckpt'),
            global_step=in_global_step
        )
        print("模型已保存到:", save_path)

    def train_step(self, positions, probs, winners, learning_rate):
        # 执行一次训练步骤
        feed_dict = {
            self.inputs_: positions,  # 输入棋盘状态
            self.training: True,  # 标记为训练模式
            self.learning_rate: learning_rate,  # 当前学习率
            self.pi_: probs,  # 策略目标（走法概率）
            self.z_: winners  # 价值目标（胜负结果）
        }

        # 运行训练操作，返回准确率、损失、当前步数和日志
        _, accuracy, loss, global_step, summary = self.sess.run(
            [self.train_op, self.accuracy, self.loss, self.global_step, self.summaries_op],
            feed_dict=feed_dict
        )
        self.train_writer.add_summary(summary, global_step)  # 写入训练日志
        return accuracy, loss, global_step

    def forward(self, positions):
        # 前向传播：给定棋盘状态，返回策略概率和价值估计
        positions = np.array(positions)  # 转成numpy数组
        # 按GPU数量拆分批次（确保每个GPU负载均匀）
        batch_n = positions.shape[0] // self.num_gpus  # 每个GPU的样本数
        alone = positions.shape[0] % self.num_gpus  # 不能均分的剩余样本数

        if alone != 0:
            # 样本数不能被GPU数量整除时的处理
            if positions.shape[0] != 1:
                # 先处理能均分的部分
                feed_dict = {
                    self.inputs_: positions[:positions.shape[0] - alone],
                    self.training: False  # 推理模式，不启用 dropout 等
                }
                action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
                # 拼接多个GPU的输出
                action_probs, value = np.vstack(action_probs), np.vstack(value)

            # 处理剩余样本（通过重复样本凑够GPU数量的倍数）
            new_positions = positions[positions.shape[0] - alone:]
            pos_lst = []
            # 凑够能被GPU数量整除的样本数
            while len(pos_lst) == 0 or (np.array(pos_lst).shape[0] * np.array(pos_lst).shape[1]) % self.num_gpus != 0:
                pos_lst.append(new_positions)
            # 调整形状，确保维度正确
            if len(pos_lst) != 0:
                shape = np.array(pos_lst).shape
                pos_lst = np.array(pos_lst).reshape([shape[0] * shape[1], 9, 10, 14])

            # 推理剩余样本
            feed_dict = {
                self.inputs_: pos_lst,
                self.training: False
            }
            action_probs_2, value_2 = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
            action_probs_2, value_2 = action_probs_2[0], value_2[0]  # 取第一个GPU的结果

            # 拼接两部分结果
            if positions.shape[0] != 1:
                action_probs = np.concatenate((action_probs, action_probs_2), axis=0)
                value = np.concatenate((value, value_2), axis=0)
                return action_probs, value
            else:
                return action_probs_2, value_2
        else:
            # 样本数能被GPU数量整除，直接推理
            feed_dict = {
                self.inputs_: positions,
                self.training: False
            }
            action_probs, value = self.sess.run([self.policy_head, self.value_head], feed_dict=feed_dict)
            return np.vstack(action_probs), np.vstack(value)  # 拼接多个GPU的输出