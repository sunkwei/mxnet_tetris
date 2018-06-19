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
from train_rnn import build_net, Batch, MAX_SEQ_LENGTH
import mxnet as mx


curr_path = osp.dirname(osp.abspath(__file__))
MODE = 'train'        # 如果生成键盘序列，作为 "train" 集合，还是 "val" 集合？
START_LEVEL = 0       # 从 8 开始，方便快速生成样本
EPOCH = 0 # 当自动时,使用的 epoch
AUTO_USING_RULE = True # 使用规则 auto_xxx()


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
        self.level_intervals_ = [1.0, 0.75, 0.6, 0.55, 0.5, 0.475, 0.450, 0.4, 0.375, 0.35, 0.33, 0.3, 0.25, 0.2, 0.1] #
        self.data_ = [[0 for i in range(self.cols_)] for i in range(self.rows_)]  # 一个二维数组标识当前数据 ...
        self.shape_ = None
        self.quit_ = False
        self.pause_ = False
        self.show_ = Show()
        self.auto_ = autoplay
        self.reset_cnt_ = 0
        if self.auto_:
            self.blurred_ = False
            if not AUTO_USING_RULE:
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

        if MODE == 'online' and autoplay:
            self.prepare_online_train()

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
        elif AUTO_USING_RULE:
            return 0xff & self.show_.wait_key(10)
        else:
            return self.pred_seq()

    def pred_seq(self):
        ''' 循环预测, 直到返回6, 总是返回 32
        '''
        for i in range(MAX_SEQ_LENGTH):
            k = self.pred_.pred(self.data2np())
            if k == 6:
                return 32  # 结束
            
            if k == 0 or k == 1 or k == 5:
                continue

            if k == 2:
                poss = self.shape_.left()
            elif k == 3:
                poss = self.shape_.rotate()
            elif k == 4:
                poss = self.shape_.right()
            
            if self.can_move(poss):
                self.move(poss)
            
            self.show_.wait_key(500)

        print('oooooh, cannot got space, JUST do it')
        return 32
                
    def interact(self):
        if self.auto_:
            if AUTO_USING_RULE:
                self.interact0(0.01)
            else:
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
                    self.move(poss)
            elif key == 32:
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
        self.shape_ = self.factory.create(int(self.cols_/2-1))
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
        if self.auto_:
            if not AUTO_USING_RULE:
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
                if self.save_rnn_:
                    # 清空重来 ...
                    self.data_ = [[0 for i in range(self.cols_)] for i in range(self.rows_)]  # 一个二维数组标识当前数据 ...
                    self.shape_ = None
                    self.reset_cnt_ += 1
                    print('GAME OVER, RESET ALL {} times'.format(self.reset_cnt_))
                    self.eliminate_rows_ = 0
                else:
                    self.over_ = True
                return
            
            if self.auto_ and AUTO_USING_RULE:
                # 此时找到最佳位置
                self.save_all()
                pos = self.select_best_pos()
                keys = self.get_key_seq(self.shape_.poss(), pos)
                keys.append(32) # 追加一个空格...
                self.restore_all()
                self.save_image_keys(keys)
                self.move(pos)

        # 当自动保存样本时，如果最高行超过10行，则强制清空所有
        # if self.save_rnn_:
        #     # if self.auto_get_empty_lines(self.data2np()) < self.rows_ / 2:
        #     # 为了生成初始状态的样本，总是尽快的清空，当消除行后，立即清空，然后从头再来
        #     if self.eliminate_rows_ > 1:
        #         self.data_ = [[0 for i in range(self.cols_)] for i in range(self.rows_)]  # 一个二维数组标识当前数据 ...
        #         self.shape_ = None
        #         self.reset_cnt_ += 1
        #         print('RESET ALL {} times'.format(self.reset_cnt_))
        #         self.eliminate_rows_ = 0
        #         return


        # 处理键盘事件    
        self.interact()
        if self.quit_:
            return
        # 超时下落
        if not self.pause_:
            poss = self.shape_.down()
            if self.can_move(poss):
                self.move(poss)
            else:
                # 当前形状已经无法移动, 准备生成下一个 ...
                self.shape_ = None
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
            #self.rnn_ops_.append((self.data2np(), 1))
            if len(self.rnn_ops_) >= MAX_SEQ_LENGTH:
                print('WARN: the rnn seq tooooo long !!!')
                return

            if MODE == 'online':
                if self.blurred_:
                    self.blurred_ = False  # 不要学习随机出错的 ..
                    print('... skip')
                else:
                    self.online_train_step(self.rnn_ops_)
            else:
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
    def get_key_seq(self, curr_poss, target_poss):
        ''' 找出, 从 curr_poss 到 target_poss 需要的按键序列
                通过 curr_poss[0], [1] 的方向,和 target_poss[0], [1] 的方向计算需要旋转的次数
                然后通过 target_poss[0] 与 curr_poss[0] 的位移, 计算平移的次数
                这里不会有下降的 ...
        '''
        key_seq = []

        poss0 = self.shape_.poss()

        curr_v = (curr_poss[1][0] - curr_poss[0][0], curr_poss[1][1] - curr_poss[0][1])
        tar_v = (target_poss[1][0] - target_poss[0][0], target_poss[1][1] - target_poss[0][1])

        for i in range(4):
            if curr_v[0] == tar_v[0] and curr_v[1] == tar_v[1]:
                break
            key_seq.append(UP)  # 旋转按键
            poss = self.shape_.rotate()
            self.move(poss)
            curr_v = (poss[1][0] - poss[0][0], poss[1][1] - poss[0][1])

        curr_poss = self.shape_.poss()
        dx = target_poss[0][1] - curr_poss[0][1]
        if dx > 0:
            for i in range(dx):
                key_seq.append(RIGHT)
        elif dx < 0:
            for i in range(-dx):
                key_seq.append(LEFT)

        self.shape_.set_poss(poss0)
        return key_seq
        
    def save_image_keys(self, keys_seq):
        ''' 根据 keys 生成 images, 保存, 用于生成训练样本
        '''
        keys = {LEFT:2, UP:3, RIGHT:4, DOWN:5, 32:6}
        self.save_rnn_begin()
        for k in keys_seq:
            self.save_rnn_key(k)    # 保存当前图像 + 对应的按键

            # 根据按键, 进行移动
            if k == LEFT:
                poss = self.shape_.left()
                self.move(poss, showing=False)
            elif k == UP:
                poss = self.shape_.rotate()
                self.move(poss, showing=False)
            elif k == RIGHT:
                poss = self.shape_.right()
                self.move(poss, showing=False)
            elif k == DOWN:
                assert(k != DOWN)
        self.save_rnn_end()


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

        # 使用权重吧, holes * 3.0 + (self.rows_ - elims) * 2.0 + (self.rows_ - els) * 1.0 + mse * .01
        rs = [ h*15.0 - e*1.0 + n*1.5 + m*5. for h,e,n,m in all_down_results ]
        rs = np.array(rs)
        idx = np.argmin(rs)
        idx_min = np.where(rs == rs[idx])[0]
        idx = random.randint(0, len(idx_min)-1)
        best_idx = idx_min[idx].item()
        
        # print('==> v={}, h:{}, elims:{}, els:{}, mse:{}'.format(
        #         rs[idx],
        #         all_down_results[best_idx][0],
        #         all_down_results[best_idx][1], all_down_results[best_idx][2],
        #         all_down_results[best_idx][3]))

        # XXX: 为了制造点混乱，需要偶尔随机选择一次
        # if self.shapes_ % 11 == 9:
        #     x = random.randint(0, len(all_start_poss)-1)
        #     print('blurring from {} to {}'.format(best_idx, x))
        #     best_idx = x
        #     self.blurred_ = False # 不要学习这个 ..
        return all_start_poss[best_idx]

    def auto_get_all_start_poss(self):
        ''' 生成当前水平移动 + 旋转的所有位置, 返回:
                1. 目标位置
        '''
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
        
        return (self.auto_get_holes(data), self.auto_get_elims(data), 
                self.auto_get_neighbor_delta(data), self.auto_get_mse(data))

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

    def auto_get_mse(self, d):
        ''' 每列从低向上看, 根据高度计算 mse, 要尽量保证 mse 比较小
        '''
        cs = np.split(d, self.cols_, axis=1)
        cs = [ c.reshape((self.rows_,)) for c in cs ]
        hs = [ np.nonzero(c)[0] for c in cs ]
        hs = [ self.rows_ - h[0].item() if len(h) > 0 else 0 for h in hs ]
        hs = np.array(hs)

        mse = np.sqrt(np.sum(np.power(hs - np.mean(hs), 2)))
        return mse.item()
                    
    def auto_get_elims(self, d):
        ''' 检查消除的行数
        '''
        rows = 0
        for r in d:
            ds = dict(zip(*np.unique(r, return_counts=True)))
            if 0 not in ds:
                rows += 1
        return rows

    def auto_get_neighbor_delta(self, d):
        ''' 获取相邻高度差的和，就是说，要尽量不出现相邻海拔差大的情况，这种情况往往只能依赖“竖棍”才能消除
        '''
        cs = np.split(d, self.cols_, axis=1)
        cs = [ c.reshape((self.rows_,)) for c in cs ]
        hs = [ np.nonzero(c)[0] for c in cs ]
        hs = [ self.rows_ - h[0].item() if len(h) > 0 else 0 for h in hs ]
        
        s = 0
        import math
        for i in range(len(hs)-1):
            s += math.fabs(hs[i+1] - hs[i])

        return s
        
    def auto_get_empty_lines(self, d):
        ''' 返回完整的空行数
        '''
        assert(d.shape == (self.rows_, self.cols_))
        emls = np.sum(d, axis=1)
        cs = dict(zip(*np.unique(emls, return_counts=True)))
        if 0 in cs:
            return cs[0]
        return 0


    ############################################################
    def prepare_online_train(self):
        ''' 准备在线训练模型，总是加载 online-0000.params
        '''
        net,stack = build_net()
        self.mod_ = mx.mod.Module(net, data_names=('data',), label_names=('label',))
        self.mod_.bind(data_shapes=(('data', (1,MAX_SEQ_LENGTH,20,14)),),      
                label_shapes=(('label',(1,MAX_SEQ_LENGTH)),), for_training=True)    # batch_size = 1

        if osp.isfile(curr_path + '/online-0000.params'):
            _,args,auxs = mx.rnn.load_rnn_checkpoint(stack, curr_path + '/online', 0)
            self.mod_.set_params(args, auxs)
            print('====resume OK')
        else:
            init = mx.init.Xavier(factor_type='in', magnitude=2.34)
            self.mod_.init_params(init)
        
        self.mod_.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate',0.0001), ('momentum',0.9)))
        self.online_train_cnt_ = 0
        self.online_train_stack_ = stack
        self.online_train_net_ = net


    def to_show_array(self, arr):
        ''' 将 arr 转化为 list，并且从后删除所有 0 '''
        if isinstance(arr, np.ndarray):
            arr = arr.reshape(-1).astype(np.int32).tolist()
        # while len(arr) > 0:
        #     if arr[-1] == 0:
        #         del arr[-1]
        #     else:
        #         break
        # if not arr:
        #     arr.append(0)
        return arr


    def online_train_step(self, rnn_ops):
        # rnn_ops 为记录的当前按键序列以及对应的 img
        assert(self.mod_)

        imgs = []
        keys = []
        for op in rnn_ops:
            imgs.append(op[0].reshape((1, self.rows_, self.cols_)).astype(np.float32))
            keys.append(op[1])
    
        imgs += [ np.zeros((1, self.rows_, self.cols_), dtype=np.float32)] * (MAX_SEQ_LENGTH-len(keys))
        imgs = np.vstack(imgs).reshape((1, MAX_SEQ_LENGTH, self.rows_, self.cols_))
        labels = np.array(keys + [ 0 ] * (MAX_SEQ_LENGTH - len(keys))).reshape((1, MAX_SEQ_LENGTH))

        batch = Batch([mx.nd.array(imgs)], ['data'], [mx.nd.array(labels)], ['label'], fname=None)
        self.mod_.forward_backward(batch)
        self.mod_.update()

        self.online_train_cnt_ += 1

        if self.online_train_cnt_ % 10 == 9:
            # 打印训练预测输出
            outss = self.mod_.get_outputs()
            outs = [ np.argmax(o.asnumpy()) for o in outss]
            label = batch.label[0].asnumpy()
            pred = outs
            print('======= {} ======'.format(self.online_train_cnt_+1))
            print('    label:{}'.format(self.to_show_array(label)))
            print('    pred: {}'.format(self.to_show_array(pred)))
            
        if self.online_train_cnt_ % 100 == 99:
            # 保存 checkpoint
            print('saveing checkpoint')
            args,auxs = self.mod_.get_params()
            mx.rnn.save_rnn_checkpoint(self.online_train_stack_, curr_path+'/online', 0, 
                    self.online_train_net_, args, auxs)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == 'val':
        MODE = 'val'    # 生成校验样本
        autoplay = True
        save_rnn = True
    elif len(sys.argv) == 2 and sys.argv[1] == 'train':
        MODE = 'train'  # 生成训练样本
        autoplay = True
        save_rnn = True
    elif len(sys.argv) == 2 and sys.argv[1] == 'online':
        MODE = 'online' # 在线训练
        autoplay = True
        save_rnn = True
    else:
        MODE = 'test'
        autoplay = True
        save_rnn = False
        AUTO_USING_RULE = False

    game = Game(autoplay=autoplay, save_rnn=save_rnn)

    while not game.gameover():
        game.step()
        if game.quit():
            break
    game.wait_quit()
    
