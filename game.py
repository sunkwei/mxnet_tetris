#!/bin/env python
#coding: utf-8

import time
import random
from show import Show
from shape import ShapeFactory
import copy
import os.path as osp
from collections import namedtuple
import numpy as np



curr_path = osp.dirname(osp.abspath(__file__))
MODE = 'train'        # 如果生成键盘序列，作为 "train" 集合，还是 "val" 集合？
START_LEVEL = 1       # 从 8 开始，方便快速生成样本
EPOCH = 29 # 当自动时,使用的 epoch
TRY = True # 使用规则 auto_xxx()


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


class Game:
    def __init__(self, autoplay=False, save_rnn=False):
        ''' 决定行列数 ...
        '''
        self.rows_ = 20
        self.cols_ = 14
        self.factory = ShapeFactory()
        self.over_ = False
        self.state_ = 0
        self.shapes_ = 0
        self.curr_level_ = START_LEVEL
        self.eliminate_rows_ = 10*self.curr_level_ # :)
        self.level_intervals_ = [1.0, 0.75, 0.6, 0.55, 0.5, 0.475, 0.450, 0.4, 0.375,] #
        self.data_ = [[0 for i in range(self.cols_)] for i in range(self.rows_)]  # 一个二维数组标识当前数据 ...
        self.shape_ = None
        self.quit_ = False
        self.pause_ = False
        self.show_ = Show()
        self.auto_ = autoplay
        if self.auto_:
            if not TRY:
                from inference_rnn import Inference
                self.pred_ = Inference(epoch=EPOCH)    # 加载 rnn 模型进行预测 ...
        
        #准备记录键盘操作序列
        self.save_rnn_ = save_rnn
        if save_rnn:
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

    def move(self, poss, showing=True):
        # 移动 self.shape_ 到新的位置poss
        old_poss = self.shape_.poss()

        for p in old_poss:
            self.data_[p[0]][p[1]] = 0  # 旧位置清空
        for p in poss:
            self.data_[p[0]][p[1]] = self.shape_.d()
        self.shape_.set_poss(poss)
        
        if showing:
            self.show_.show(self.data_)

    def wait_key(self, timeout):
        timeout = int(timeout*1000)
        if timeout <= 0:
            timeout = 1
        if not self.auto_:
            return 0xff & self.show_.wait_key(timeout)
        elif TRY:
            return 0xff & self.show_.wait_key(50)
        else:
            # 使用 self.pred_ 进行预测 ...
            img = self.data2np()
            key = self.pred_.pred(img)
            keys = {
                0: 0,           # 0, 1 将被忽略 ...
                1: 1, 
                2: LEFT,
                3: UP,
                4: RIGHT,
                5: DOWN,
                6: 32,
            }
            self.show_.wait_key(timeout)
            if key == 5:
                key = 0
            return keys[key]

    def interact(self):
        if self.auto_:
            if TRY:
                self.interact0(0.2)
            else:
                # FIXME: 因为在一行上，可能联系平移，这里给多次机会吧
                for i in range(1):
                    self.interact0(0.5, once=True)
        else:
            self.interact0(self.level_intervals_[self.curr_level_])

    def interact0(self, level_time, once=False):
        t0 = time.time()
        t = t0
        n = 2
        while n > 0:
            n -= 1
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
                poss = keymaps[key]()
                if self.can_move(poss):
                    self.save_rnn_key(key)
                    self.move(poss)
            elif key == 32:
                self.save_rnn_key(32)
                poss = self.shape_.down()
                while self.can_move(poss):
                    self.move(poss)
                    poss = self.shape_.down()
                break
            elif key == 27:
                self.quit_ = True
                break
            elif key == ord('P') or key == ord('p'):
                self.pause_ = not self.pause_
            elif key == ord('B') or key == ord('b'):
                self.bomb()
            t = time.time()
            if once:
                break

    def bomb(self):
        ''' 直接消掉地下十行
        '''
        del self.data_[self.rows_-10:self.rows_]
        for i in range(10):
            self.data_.insert(0, [0 for i in range(self.cols_)])

    def create_new_shape(self):
        ''' 如果空间不够，返回 False
        '''
        self.shape_ = self.factory.create(self.cols_/2-1)
        poss = self.shape_.poss()
        # poss 占用的空间不能有非 0
        empty = True
        for p in poss:
            if self.data_[p[0]][p[1]] != 0:
                empty = False
                break
        if not empty:
            return False
        self.move(poss)
        self.shapes_ += 1
        self.save_rnn_begin()
        if self.auto_:
            if not TRY:
                self.pred_.reset()
        return True

    def step(self):
        ''' 每个 step 对应 self.level_intervals_[self.curr_level_] 的等待时间, 
            这段时间内响应键盘事件, 更新显示界面
        '''
        self.show_.show_info("L:{},rs:{},i:{},ss:{},{}".format(self.curr_level_, 
                self.eliminate_rows_, self.level_intervals_[self.curr_level_], 
                self.shapes_, 'pause' if self.pause_ else 'playing'))
        if self.shape_ is None:
            # 新的形状 ..
            if not self.create_new_shape():
                self.over_ = True
                return
            
            if self.shapes_ == 33:
                aa = 0
            if self.auto_ and TRY:
                # 此时找到最佳位置
                pos = self.select_best_pos()
                self.move(pos)

        # 处理键盘事件    
        self.interact()
        if self.quit_:
            return
        # 超时下落
        if not self.pause_:
            poss = self.shape_.down()
            if self.can_move(poss):
                self.save_rnn_key(DOWN)
                self.move(poss)
            else:
                # 当前形状已经无法移动, 准备生成下一个 ...
                self.shape_ = None
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
        if self.save_rnn_:
            self.rnn_ops_ = []
            # 以后生成的不再包含 0, 方便 fix_label.py 工具识别
            # self.rnn_ops_.append((self.data2np(), 0))

    def save_rnn_end(self):
        ''' 当一个形状无法活动时 ...
            将 rnn_ops_ 中的操作合并
        '''
        if self.save_rnn_:
            self.rnn_ops_.append((self.data2np(), 1))
            if len(self.rnn_ops_) >= 50:
                print('WARN: the rnn seq tooooo long !!!')
                return
            fname = '{}{}.npz'.format(self.rnn_fname_prefix, self.shapes_)
            ds = [ r[0] for r in self.rnn_ops_ ]
            ks = [ r[1] for r in self.rnn_ops_ ]
            np.savez_compressed(fname, imgs=np.stack(ds), keys=np.array(ks))
            print('save rnn sample: {}'.format(fname))

    def save_rnn_key(self, key):
        ''' 保存按键，仅仅四个，上下左右，特别的 "自由下落" 和 "快速下落" 都映射为 down 了,
            加上 "开始", "结束":
                0: 开始
                1: 结束
                2: left
                3: up
                4: right
                5: down
                6: 空格，快速下降
        '''
        if self.save_rnn_:
            keys = {LEFT:2, UP:3, RIGHT:4, DOWN:5, 32:6}
            assert(key in keys)
            self.rnn_ops_.append((self.data2np(), keys[key]))

    #########################################################################################
    def save_all(self):
        ''' 保存当前 self.data_, self.shape_, ...
        '''
        self.saved_data_ = copy.deepcopy(self.data_)
        self.saved_shape_ = copy.deepcopy(self.shape_)

    def restore_all(self):
        self.data_ = copy.deepcopy(self.saved_data_)
        self.shape_ = copy.deepcopy(self.saved_shape_)

    def select_best_pos(self):
        ''' XXX: 为 self.shape_ 选择一个最佳目标位置, 下落后, 保证:
                1. 无"空洞"
                2. 多消除行
                3. 总高度最低
                4. 每列最高点的方差应该尽量小, 就是说, 应该让每列高度更平均

            FIXME: 采用最暴力的穷举, 从左到右, 旋转三次, 看每次下落的结果, 评估上面的指标
        '''
        self.save_all()
        
        if self.shape_.d() == 2:
            aa = 0
        all_start_poss = self.auto_get_all_start_poss()
        all_down_results = [ self.auto_get_down_result(poss) for poss in all_start_poss ]
        assert(len(all_down_results) == len(all_start_poss))

        self.restore_all()

        # 根据 result 找出最合理的 start_poss
        # 先扔掉"洞"多的,再找消除行多的,最后找空行最多的
        results = np.array(all_down_results)
        
        holes = results[:,0]
        min_h = np.argmin(holes) # 从第一列中找出"洞"最少的
        idx_holes = np.where(results[:,0] == holes[min_h])
        idx_holes = idx_holes[0]
        elims = results[idx_holes]    # 这些行中, 第一列对应"洞"最少的, 从中查找

        max_elims = np.argmax(elims[:,1])  # 从第二列中找出消除最多的行数
        idx_elims = np.where(elims[:,1] == elims[:,1][max_elims])
        idx_elims = idx_elims[0]
        els = elims[idx_elims]      # 这些行对应着"洞最少",并且消除行最多的

        max_els = np.argmax(els[:,2]) # 从第三列找出保留空行最多的
        idx_els = np.where(els[:,2] == els[:,2][max_els])
        idx_els = idx_els[0]

        best_idx = idx_holes[idx_elims][idx_els]
        best_idx = best_idx.tolist()[0]

        print('best pos: {}, I:{}/{}/{}'.format(all_start_poss[best_idx], 
                holes[min_h], elims[:,1][max_elims], els[:,2][max_els]))
        
        return all_start_poss[best_idx]

    def auto_get_all_start_poss(self):
        ''' 生成当前水平移动 + 旋转的所有位置, 返回列表 '''
        all_poss = []

        poss0 = self.shape_.poss()
        for i in range(4):
            ''' 四次旋转后, 每次移动到最左侧, 然后依次右移, 记录允许的启动位置 '''
            self.move(poss0)
            for n in range(i):
                poss = self.shape_.rotate()
                if not self.can_move(poss):
                    continue
                self.move(poss)

            # 移动到最左侧
            poss = self.shape_.left()
            while self.can_move(poss):
                self.move(poss, showing=False)
                poss = self.shape_.left()

            # 依次右移, 记录每个位置
            poss = self.shape_.poss()    # XXX: 这样只是为了方便 ...
            while self.can_move(poss):
                self.move(poss, showing=False)
                if poss not in all_poss:
                    all_poss.append(poss)

                poss = self.shape_.right()
                
        return all_poss

    def auto_get_down_result(self, poss):
        ''' 从 poss 作为其实点, 下降直到无法下降后, 返回 (holes, elims, emls)   (空洞数, 消除行数, 完整空行数)
        '''
        assert(self.can_move(poss))
        self.move(poss)
        
        poss = self.shape_.down()
        while self.can_move(poss):
            self.move(poss, showing=False)
            poss = self.shape_.down()
        data = self.data2np()
        
        return (self.auto_get_holes(data), self.auto_get_elims(data), self.auto_get_empty_lines(data))

    def auto_get_holes(self, d):
        ''' 从 self.data_ 最低开始, 水平找到 0 后, 向上找, 如果右非 0 则认为是空洞
        '''
        holes = 0
        cs = np.split(d, self.cols_, axis=1)    # cs 为每列
        for c in cs:
            found = False
            n = 0
            c = c.reshape((self.rows_,)).tolist()
            for i in range(len(c)):
                if c[i]:
                    found = True
                    continue
                if found and c[i] == 0:
                    n += 1
            holes += n
        return holes
                    
    def auto_get_elims(self, d):
        ''' 检查消除的行数
        '''
        rows = 0
        for r in d:
            ds = dict(zip(*np.unique(r, return_counts=True)))
            if 0 not in ds:
                rows += 1
        return rows
        
    def auto_get_empty_lines(self, d):
        ''' 返回完整的空行数
        '''
        assert(d.shape == (self.rows_, self.cols_))
        emls = np.sum(d, axis=1)
        cs = dict(zip(*np.unique(emls, return_counts=True)))
        if 0 in cs:
            return cs[0]
        return 0



if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'val':
        MODE = 'val'    # 生成校验样本
        autoplay = False
        save_rnn = True
    elif len(sys.argv) == 2 and sys.argv[1] == 'train':
        MODE = 'train'  # 生成训练样本
        autoplay = False
        save_rnn = True
    else:
        MODE = 'test'
        autoplay = True
        save_rnn = False
    game = Game(autoplay=autoplay, save_rnn=save_rnn)
    while not game.gameover():
        game.step()
        if game.quit():
            break
    game.wait_quit()
    
