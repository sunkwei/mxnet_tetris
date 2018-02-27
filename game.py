#!/bin/env python
#coding: utf-8

import time
import random
from show import Show
from shape import ShapeFactory
import copy
import os.path as osp
from db import DB
import mxnet as mx
import numpy as np
from collections import namedtuple
        #

curr_path = osp.dirname(osp.abspath(__file__))


# 方向按键对应的数值
LEFT = 81
UP = 82
RIGHT = 83
DOWN = 84

KEYS = {
    32: 0,      # 空格
    LEFT: 1,    # 方向
    UP: 2,
    RIGHT: 3,
    DOWN: 4,    # 下降
    -1: 5,      # 自由下降, 其实 0,4,5 可以视为一类, 都是不再需要修正方向的
}

RNN = True  # 记录一个形状出现到结束的完整序列 ...
MODE = 'train'

class Game:
    def __init__(self, save_oper=False, auto=False, rnn=RNN):
        ''' 决定行列数 ...
        '''
        self.rows_ = 20
        self.save_oper_ = save_oper
        self.cols_ = 14
        self.eliminate_rows_ = 0
        self.factory = ShapeFactory()
        self.over_ = False
        self.curr_level_ = 0
        self.state_ = 0
        self.shapes_ = 0
        self.level_intervals_ = [1.0, 0.75, 0.6, 0.55, 0.5, 0.475, 0.450, 0.4, 0.375, 0.35, 0.3, 0.275, 0.25, 0.2] #
        self.data_ = [[0 for i in range(self.cols_)] for i in range(self.rows_)]  # 一个二维数组标识当前数据 ...
        self.shape_ = None
        self.quit_ = False
        self.pause_ = False
        self.show_ = Show()
        if self.save_oper_:
            self.db_ = DB(table='train') # 用于记录按键操作, 保存训练样本
        self.auto_ = auto
        if self.auto_:
            self.load_model()
        self.rnn_ = rnn
        if self.rnn_:
            try:
                import os
                os.mkdir(curr_path+'/rnn_{}'.format(MODE))
            except:
                pass
            self.rnn_fname_prefix = curr_path+'/rnn_{}'.format(MODE)+'/'+time.strftime('%j%H%M-', time.localtime())
            self.rnn_ops_ = []  # 记录操作序列，[(np(data_), key), ... ]
                            # np(data_) 为当前 self.data_ 的np数组，key 为对当前的操作

    def gameover(self):
        return self.over_

    def data2np(self):
        ''' 将 self.data_ 转化为 numpy 数组 '''
        rs = [ np.array(r) for r in self.data_ ]
        return np.vstack(rs)

    def quit(self):
        return self.quit_

    def load_model(self):
        ''' 加载模型
        '''
        sym,args,auxs = mx.model.load_checkpoint(osp.sep.join((curr_path, 'mx')), 1)
        self.mod_ = mx.mod.Module(sym, data_names=('data',), label_names=None)
        self.mod_.bind(data_shapes=(('data',(1,1,self.rows_,self.cols_)),), for_training=False)
        self.mod_.set_params(args, auxs, allow_missing=True)

    def isme(self, p, poss):
        ''' 目标位置是否还是属于自己
        '''
        for pos in poss:
            if pos[0] == p[0] and pos[1] == p[1]:
                return True
        return False

    def can_move(self, poss):
        # 检查 poss 位置是否有效, 坐标必须在有效范围, 并且self.data_[pos] == 0
        for p in poss:
            if p[0] < 0 or p[1] < 0 or p[0] >= self.rows_ or p[1] >= self.cols_:
                return False
            # 自己形状内的移动不受限制
            if self.data_[p[0]][p[1]] != 0 and not self.isme(p, self.shape_.poss()):
                return False
        return True

    def move(self, poss):
        # 移动 self.shape_ 到新的位置poss
        old_poss = self.shape_.poss()
        for p in old_poss:
            self.data_[p[0]][p[1]] = 0  # 旧位置清空
        for p in poss:
            self.data_[p[0]][p[1]] = self.shape_.d()
        self.shape_.set_poss(poss)
        self.show_.show(self.data_)

    def wait_key(self, timeout):
        timeout = int(timeout*1000)
        if timeout <= 0:
            timeout = 1
        if not self.auto_:
            return 0xff & self.show_.wait_key(timeout)
        else:
            rs = [ np.array(r) for r in self.data_ ]
            data = np.vstack(rs)
            Batch = namedtuple('Batch', ['data',])
            data = mx.nd.array(data.reshape(1,1,self.rows_,self.cols_))
            batch = Batch(data=[data])
            self.mod_.forward(batch)
            out = self.mod_.get_outputs()[0].asnumpy()
            code = np.argmax(out, axis=1)[0]
            scale = out[0][code]
            keys = KEYS.keys()
            key = keys[code]
            self.show_.wait_key(timeout)
            return key

    def interact(self):
        if self.auto_:
            # FIXME: 因为在一行上，可能联系平移，这里给三次机会吧
            for i in range(3):
                self.interact0(0.0001)
        else:
            self.interact0(self.level_intervals_[self.curr_level_])

    def interact0(self, level_time):
        t0 = time.time()
        t = t0
        while True:
            elapse = t - t0  # 已经等待的时间
            if elapse >= level_time:
                break
            wait = level_time - elapse
            key = self.wait_key(wait)
            keymaps = {
                LEFT: self.shape_.left,
                UP: self.shape_.rotate,
                RIGHT: self.shape_.right,
                DOWN: self.shape_.down,
            }
            if key in keymaps:
                self.save_oper(key)
                poss = keymaps[key]()
                if self.can_move(poss):
                    self.move(poss)
                    if self.rnn_:
                        self.save_rnn_key(key)
            elif key == 32:
                self.save_oper(key)
                poss = self.shape_.down()
                while self.can_move(poss):
                    self.move(poss)
                    if self.rnn_:
                        self.save_rnn_key(DOWN)
                    poss = self.shape_.down()
                if self.save_oper_:
                    break
            elif key == 27:
                self.quit_ = True
                break
            elif key == ord('P') or key == ord('p'):
                self.pause_ = not self.pause_
            elif key == ord('B') or key == ord('b'):
                self.bomb()
            t = time.time()

    def bomb(self):
        ''' 直接消掉地下十行
        '''
        del self.data_[self.rows_-10:self.rows_]
        for i in range(10):
            self.data_.insert(0, [0 for i in range(self.cols_)])

    def save_oper(self, key):
        ''' 保存 self.data_ 和按键, 对应的:
                按键作为 "标签", data_ 作为 "img"
                希望通过大量的学习, 能训练一个自动玩的网络,
                该网络输入 self.data_ 的数据, 输出为按键

            将数据保存到 sqlite3 中
        '''
        if not self.save_oper_:
            return
        if key in KEYS:
            s = ''
            for r in self.data_:
                s += ','.join(str(x) for x in r)
                s += ','
            self.db_.save(self.rows_, self.cols_, s, KEYS[key], 1)

    def step(self):
        ''' 每个 step 对应 self.level_intervals_[self.curr_level_] 的等待时间, 
            这段时间内响应键盘事件, 更新显示界面
        '''
        self.show_.show_info("L:{},rs:{},i:{},ss:{},{}".format(self.curr_level_, 
                self.eliminate_rows_, self.level_intervals_[self.curr_level_], 
                self.shapes_, 'pause' if self.pause_ else 'playing'))
        if self.shape_ is None:
            self.shape_ = self.factory.create(self.cols_/2-1)
            poss = self.shape_.poss()
            # poss 占用的空间不能有非 0
            empty = True
            for p in poss:
                if self.data_[p[0]][p[1]] != 0:
                    empty = False
                    break
            if not empty:
                self.over_ = True
                return
            self.move(poss)
            self.shapes_ += 1
            if self.rnn_:
                self.save_rnn_begin()
        # 处理键盘事件    
        self.interact()
        if self.quit_:
            return
        # 超时下落
        if not self.pause_:
            poss = self.shape_.down()
            if self.can_move(poss):
                self.move(poss)
                if self.rnn_:
                    self.save_rnn_key(DOWN)
            else:
                # 当前形状已经无法移动, 准备生成下一个 ...
                self.shape_ = None
                if self.rnn_:
                    self.save_rnn_end()
                # 清除满行
                self.eliminate_rows_ += self.clear_rows()
                self.curr_level_ = self.eliminate_rows_ // 10   # 每十行增加一级 ...
                if self.curr_level_ >= len(self.level_intervals_):
                    self.curr_level_ = len(self.level_intervals_) - 1

    def clear_rows(self):
        ''' 消除满行, 操作 self.data_, 从后往前, 删除满行, 并且从头插入新的空行
            返回删除的满行数
        '''
        full_row_ids = []
        for i,r in enumerate(self.data_):
            f = True
            for c in r:
                if c == 0:
                    f = False
                    break
            if f:
                full_row_ids.insert(0, i)   # full_row_ids 保存 self.data_ 需要删除的行的序号的倒序
        for i in full_row_ids:
            del self.data_[i]
        for i in range(len(full_row_ids)):
            self.data_.insert(0, [0 for c in range(self.cols_)])   # 插入新空行
        return len(full_row_ids)

    def wait_quit(self):
        # 当 gameover 后, 等待 q/esc 结束
        self.show_.show_gameover()
        while True:
            key = self.show_.wait_key(0)
            if key == 27:
                break

    def save_rnn_begin(self):
        ''' 对应一个新的形状产生 ...
        '''
        self.rnn_ops_ = []
        self.rnn_ops_.append((self.data2np(), 0))

    def save_rnn_end(self):
        ''' 当一个形状无法活动时 ...
            将 rnn_ops_ 中的操作合并
        '''
        self.rnn_ops_.append((self.data2np(), 1))
        if len(self.rnn_ops_) >= 50:
            print('WARN: the rnn seq tooooo long !!!')
            return
        fname = '{}{}.npz'.format(self.rnn_fname_prefix, self.shapes_)
        ds = [ r[0] for r in self.rnn_ops_ ]
        ks = [ r[1] for r in self.rnn_ops_ ]
        np.savez_compressed(fname, imgs=np.stack(ds), keys=np.array(ks))

    def save_rnn_key(self, key):
        ''' 保存按键，仅仅四个，上下左右，特别的 "自由下落" 和 "快速下落" 都映射为 down 了,
            加上 "开始", "结束":
                0: 开始
                1: 结束
                2: left
                3: up
                4: right
                5: down
        '''
        keys = {81:2, 82:3, 83:4, 84:5}
        assert(key in keys)
        self.rnn_ops_.append((self.data2np(), keys[key]))


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'val':
        MODE = 'val'    # 生成校验样本
        
    game = Game(auto=False)
    while not game.gameover():
        game.step()
        if game.quit():
            break
    game.wait_quit()
    