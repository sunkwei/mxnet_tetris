#!/bin/env python
#coding: utf-8

import numpy as np
import mxnet as mx
import os
import os.path as osp
import random
from collections import namedtuple


curr_path = osp.dirname(osp.abspath(__file__))
MAX_SEQ_LENGTH = 50 # FIXME: 最长按键序列，不够的补0，照理说应该使用 bucketing mode
RNN_HIDDEN_NUM = 128


class Batch:
    def __init__(self, data, datanames, label, labelnames):
        self.data = data
        self.label = label
        self.datanames_ = datanames
        self.labelnames_ = labelnames

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.datanames_, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.labelnames_, self.label)]


class DataSource:
    def __init__(self, prefix=curr_path+'/rnn_train'):
        self.fnames_ = []
        for fname in os.listdir(prefix):
            _,ext = osp.splitext(fname)
            if ext == '.npz':
                self.fnames_.append(osp.sep.join((prefix, fname)))
        random.shuffle(self.fnames_)
        
    def __iter__(self, batch_size=1):
        assert(batch_size == 1) # FIXME: 目前先做一个的吧，因为不定长，需要使用 bucket
        for fname in self.fnames_:
            imgs,keys = self.load(fname)
            yield Batch([mx.nd.array(imgs)], ['data'], [mx.nd.array(keys)], ['label'])
        random.shuffle(self.fnames_)    # 重新打乱
        raise StopIteration

    def load(self, fname):
        ''' data['imgs'].shape = (n, rows, cols), n 为序列长度
            data['keys'].shape = (n,)  

            注意：imgs[0,::] 作为 init_stats, 
                 keys[0] == 0，忽略
                 目的是训练：imgs[1,::] 希望收到 keys[1]
                           imgs[2,::] 希望收到 keys[2]
                            ...
        '''
        data = np.load(fname)
        imgs = data['imgs']
        keys = data['keys']
        n,rows,cols = imgs.shape
        assert(len(keys) == n)
        assert(n < MAX_SEQ_LENGTH)
        # padding:
        zeros = np.zeros((1, rows, cols))
        for i in range(MAX_SEQ_LENGTH-n):
            imgs = np.concatenate((imgs,zeros))
            keys = np.concatenate((keys,[0]))
        return imgs.reshape((1,MAX_SEQ_LENGTH,rows,cols)), keys.reshape((1,MAX_SEQ_LENGTH)) # 1 为 batch size


def build_net():
    ''' 构造 gru 网络，序列定长为 MAX_SEQ_LENGTH
        输入为
    '''
    data = mx.sym.Variable('data')          # (batch_size, seq_len, rows, cols)
    label = mx.sym.Variable('label')        # (batch_size, seq_len)
    #
    ''' FIXME: 这里如果使用卷积，将修改影响 axis=1 的含义，估计应该对帧进行卷积，再组合 ..
    '''
    # data = mx.sym.Convolution(data, num_filter=32, kernel=(4,4), pad=(1,1))
    # data = mx.sym.BatchNorm(data)
    # data = mx.sym.Activation(data, act_type='tanh')
    # data = mx.sym.Convolution(data, num_filter=64, kernel=(3,3), stride=(2,2))
    # data = mx.sym.Activation(data, act_type='tanh')
    #
    slices = mx.sym.split(data, num_outputs=MAX_SEQ_LENGTH, axis=1) # 切分帧
    data = [ s for s in slices]
    stack = mx.rnn.SequentialRNNCell()
    for i in range(3):
        stack.add(mx.rnn.GRUCell(RNN_HIDDEN_NUM, 'gru_{}_'.format(i)))
    rnn_outputs, rnn_states = stack.unroll(MAX_SEQ_LENGTH, data)
    # 构造 MAX_SEQ_LENGTH 个 softmaxout, 对应每个 rnn_outputs
    labels = mx.sym.split(label, MAX_SEQ_LENGTH)
    outputs = []
    for i,rnn_out in enumerate(rnn_outputs):
        fc = mx.sym.FullyConnected(rnn_out, num_hidden=6) # 0, LEFT, UP, RIGHT, DOWN, 1
        sm = mx.sym.SoftmaxOutput(fc, mx.sym.reshape(labels[i], (1,)))
        outputs.append(sm)
    return mx.sym.Group(outputs), stack


def to_show_array(arr):
    ''' 将 arr 转化为 list，并且从后删除所有 0 '''
    if isinstance(arr, np.ndarray):
        arr = arr.reshape(-1).astype(np.int32).tolist()
    while True:
        if arr[-1] == 0:
            del arr[-1]
        else:
            break
    return arr


def train():
    net, stack = build_net()
    mod = mx.mod.Module(net, data_names=('data',), label_names=('label',))
    mod.bind(data_shapes=(('data', (1,MAX_SEQ_LENGTH,20,14)),), 
            label_shapes=(('label',(1,MAX_SEQ_LENGTH)),), for_training=True)
    mod.init_params()
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate',0.01),))
    #
    ds_train = DataSource()
    ds_val = DataSource(prefix=curr_path+'/rnn_val')
    #
    for epoch in range(100):
        for i,batch in enumerate(ds_train):
            mod.forward_backward(batch)
            mod.update()
            outs = mod.get_outputs()
            outs = [ np.argmax(o.asnumpy()) for o in outs]
            if (i+1) % 100 == 0:
                print('------------------------------------------------------------------------------------------')
                print('T: #{:>02}/{:>05} labels={}'.format(epoch, i, to_show_array(batch.label[0].asnumpy())))
                print('              :pred={}'.format(to_show_array(outs)))
        # eval
        print('EVAL: ====================================================================')
        for i,batch in enumerate(ds_val):
            if i % 10 != 0:
                continue
            mod.forward(batch, is_train=False)
            outs = mod.get_outputs()
            outs = [ np.argmax(o.asnumpy()) for o in outs]
            print('V: #{:>02}/{:>05} labels={}'.format(epoch, i, to_show_array(batch.label[0].asnumpy())))
            print('              :pred={}'.format(to_show_array(outs)))
        # save checkpoint, 
        args,auxs = mod.get_params()
        mx.rnn.save_rnn_checkpoint(stack, curr_path+'/rnn', epoch, net, args, auxs)


if __name__ == '__main__':
    # ds = DataSource()
    # for batch in ds:
    #     print(batch)

    train()