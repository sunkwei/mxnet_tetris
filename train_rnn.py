#!/bin/env python
#coding: utf-8

import numpy as np
import mxnet as mx
import os
import os.path as osp
import random
from collections import namedtuple


#random.seed(0)
#mx.random.seed(0)


curr_path = osp.dirname(osp.abspath(__file__))
MAX_SEQ_LENGTH = 15 # FIXME: 最长按键序列，不够的补0，照理说应该使用 bucketing mode
RNN_HIDDEN_NUM = 256
RNN_LAYERS_NUM = 3
OUT_NUM = 7 # 标签数目
WRAP = False
ctx = mx.cpu()


class Batch:
    def __init__(self, data, datanames, label, labelnames, fname=None):
        self.data = data
        self.label = label
        self.datanames_ = datanames
        self.labelnames_ = labelnames
        self.fname_ = fname

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.datanames_, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.labelnames_, self.label)]


def _load_npz(fname):
    ''' data['imgs'].shape = (n, rows, cols), n 为序列长度
        data['keys'].shape = (n,)  

        注意：imgs[0,::] 作为 init_stats, 
                keys[0] == 0，忽略
                目的是训练：imgs[1,::] 希望收到 keys[1]
                        imgs[2,::] 希望收到 keys[2]
                        ...

        但是： XXX???
    '''
    data = np.load(fname)
    imgs = data['imgs'].astype(np.float32)
    if WRAP:
        keys = data['keys'].tolist()
        del keys[0]
        keys.append(0)
        keys = np.array(keys)
    else:
        keys = data['keys']
    n,rows,cols = imgs.shape
    if n >= MAX_SEQ_LENGTH:
        import sys
        print('n > MAX_SEQ_LENGTH!!!, fname={}, n={}'.format(fname, n))
        return None, None
    assert(len(keys) == n)
    assert(n < MAX_SEQ_LENGTH)
    # padding:
    zeros = np.zeros((1, rows, cols))
    for i in range(MAX_SEQ_LENGTH-n):
        imgs = np.concatenate((imgs,zeros))
        keys = np.concatenate((keys,[0]))
    return imgs.reshape((1,MAX_SEQ_LENGTH,rows,cols)), keys.reshape((1,MAX_SEQ_LENGTH)) # 1 为 batch size


class DataSource:
    def __init__(self, prefix=curr_path+'/rnn_train'):
        self.prefix_ = prefix
        self.reload(prefix)

    def reload(self, prefix):
        self.fnames_ = []
        for fname in os.listdir(prefix):
            _,ext = osp.splitext(fname)
            if ext == '.npz':
                self.fnames_.append(osp.sep.join((prefix, fname)))
        random.shuffle(self.fnames_)

    def count(self):
        return len(self.fnames_)

    def __iter__(self, batch_size=1):
        assert(batch_size == 1) # FIXME: 目前先做一个的吧，因为不定长，需要使用 bucket
        for fname in self.fnames_:
            imgs,keys = _load_npz(fname)
            if imgs is None:
                continue
            # XXX: imgs 中的数据需要将非0项都改为 1 (不同形状使用不同数字标识)
            idx = imgs[:,:,:,:] > 0
            imgs[idx] = 1.0
            imgs -= 0.1
            yield Batch([mx.nd.array(imgs)], ['data'], [mx.nd.array(keys)], ['label'], fname=fname)
        self.reload(self.prefix_)
        raise StopIteration


def to_show_array(arr):
    ''' 将 arr 转化为 list，并且从后删除所有 0 '''
    if isinstance(arr, np.ndarray):
        arr = arr.reshape(-1).astype(np.int32).tolist()
    while len(arr) > 0:
        if arr[-1] == 0:
            del arr[-1]
        else:
            break
    if not arr:
        arr.append(0)
    return arr


def levenshtein_distance(label, pred):
    n_label = len(label) + 1
    n_pred = len(pred) + 1
    if (label == pred):
        return 0
    if (len(label) == 0):
        return len(pred)
    if (len(pred) == 0):
        return len(label)

    v0 = [i for i in range(n_label)]
    v1 = [0 for i in range(n_label)]

    for i in range(len(pred)):
        v1[0] = i + 1

        for j in range(len(label)):
            cost = 0 if label[j] == pred[i] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)

        for j in range(n_label):
            v0[j] = v1[j]

    return v1[len(label)]


def build_net():
    ''' 构造 gru 网络，序列定长为 MAX_SEQ_LENGTH
        输入为
    '''
    data = mx.sym.Variable('data')          # (batch_size, seq_len, rows, cols)
    label = mx.sym.Variable('label')        # (batch_size, seq_len)
    #
    ''' FIXME: 这里如果使用卷积，将修改影响 axis=1 的含义，估计应该对帧进行卷积，再组合 ..
    '''
    frames = mx.sym.split(data, num_outputs=MAX_SEQ_LENGTH, axis=1) # 切分帧

    # conv1, conv2
    conv1_weight = mx.sym.var(name='conv1_weight')
    conv1_bias = mx.sym.var(name='conv1_bias')
    # conv2_weight = mx.sym.var(name='conv2_weight')
    # conv2_bias = mx.sym.var(name='conv2_bias')
    # conv3_weight = mx.sym.var(name='conv3_weight')
    # conv3_bias = mx.sym.var(name='conv3_bias')
    conv_outs = []
    for frame in frames:
        # frame: (batch, 1, rows, cols)
        conv1 = mx.sym.Convolution(frame, weight=conv1_weight, bias=conv1_bias,
                num_filter=16, kernel=(4,4), pad=(1,1), stride=(2,2))
        act = mx.sym.Activation(conv1, act_type='relu')
        # conv2 = mx.sym.Convolution(act, weight=conv2_weight, bias=conv2_bias,
        #         num_filter=64, kernel=(4,4), stride=(2,2))
        # act = mx.sym.Activation(conv2, act_type='tanh')
        # conv3 = mx.sym.Convolution(act, weight=conv3_weight, bias=conv3_bias,
        #         num_filter=128, kernel=(3,3), stride=(2,2))
        # act = mx.sym.Activation(conv3, act_type='relu')
        out = mx.sym.reshape(act, shape=(0,1,-1))  # 保持 batch 不变，
        conv_outs.append(out)

    data = conv_outs
    #data = [ f for f in frames ]
    
    stack = mx.rnn.SequentialRNNCell()
    for i in range(RNN_LAYERS_NUM):
        stack.add(mx.rnn.GRUCell(RNN_HIDDEN_NUM, 'gru_{}_'.format(i)))
#        stack.add(mx.rnn.Dropout(0.3))
    rnn_outputs, rnn_states = stack.unroll(MAX_SEQ_LENGTH, data)
    # 构造 MAX_SEQ_LENGTH 个 softmaxout, 对应每个 rnn_outputs
    labels = mx.sym.split(label, MAX_SEQ_LENGTH)
    outputs = []
    # 所有 fc 共享参数
    fc1_weight = mx.sym.var(name='fc1_weight')
    fc1_bias = mx.sym.var(name='fc1_bias')
    #
    fc2_weight = mx.sym.var(name='fc2_weight')
    fc2_bias = mx.sym.var(name='fc2_bias')
    #
    for i,rnn_out in enumerate(rnn_outputs):
        # XXX: 
        fc1 = mx.sym.FullyConnected(rnn_out, num_hidden=64, weight=fc1_weight, bias=fc1_bias)
        act = mx.sym.Activation(fc1, act_type='relu')
        act = mx.sym.Dropout(act, p=0.3)
        fc2 = mx.sym.FullyConnected(act, num_hidden=OUT_NUM, weight=fc2_weight, bias=fc2_bias)
        sm = mx.sym.SoftmaxOutput(fc2, mx.sym.reshape(labels[i], (1,)))
        outputs.append(sm)
    states = [ mx.sym.BlockGrad(s) for s in rnn_states ]
    #outputs.extend(states)
    return mx.sym.Group(outputs), stack



def train(resume_epoch=-1):
    net, stack = build_net()
    mod = mx.mod.Module(net, data_names=('data',), label_names=('label',), context=ctx)
    mod.bind(data_shapes=(('data', (1,MAX_SEQ_LENGTH,20,14)),), 
            label_shapes=(('label',(1,MAX_SEQ_LENGTH)),), for_training=True)
    if resume_epoch >= 0:
        sym,args,auxs = mx.rnn.load_rnn_checkpoint(stack, curr_path+'/rnn', resume_epoch)
        mod.set_params(args, auxs)
    else:
        init = mx.init.Xavier(factor_type='in', magnitude=2.34)
        mod.init_params(init)
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate',0.001), ('momentum',0.9)))
    #
    ds_train = DataSource(prefix=curr_path+'/rnn_train')
    ds_val = DataSource(prefix=curr_path+'/rnn_val')
    #
    for epoch in range(resume_epoch+1, 30):
        train_cnt = ds_train.count()
        val_cnt = ds_val.count()

        num = 0
        sum_dis = 0
        for i,batch in enumerate(ds_train):
            mod.forward_backward(batch)
            mod.update()

            outss = mod.get_outputs()
            outs = [ np.argmax(o.asnumpy()) for o in outss]
            label = batch.label[0].asnumpy()
            pred = outs
            dis = levenshtein_distance(to_show_array(label), to_show_array(pred))
            if (i+1)%(train_cnt/10) == 0:
                print('------------------------------------------------------------------------------------------')
                print('#{:>02}/{:>05} labels={}'.format(epoch, i, to_show_array(label)))
                print('           :pred={}'.format(to_show_array(pred)))
                print('        distance: {}'.format(dis))
                print('    fname={}'.format(batch.fname_))
            num += 1
            sum_dis += dis
        print('*******************************************************')
        print('TRAIN: num={}, sum_dis={}, mean_dis={}'.format(num, sum_dis, 1.0*sum_dis/num))
        eval
        print('EVAL: ====================================================================')
        num = 0
        sum_dis = 0
        for i,batch in enumerate(ds_val):
            mod.forward(batch, is_train=False)
            outs = mod.get_outputs()
            outs = [ np.argmax(o.asnumpy()) for o in outs]
            label = batch.label[0].asnumpy()
            pred = outs
            dis = levenshtein_distance(to_show_array(label), to_show_array(pred))
            if (i+1) % (val_cnt/10) == 0:
                print('#{:>02}/{:>05} labels={}'.format(epoch, i, to_show_array(label)))
                print('           :pred={}'.format(to_show_array(pred)))
                print('        distance: {}'.format(levenshtein_distance(to_show_array(label), to_show_array(pred))))
                print('     fname={}'.format(batch.fname_))
            num += 1
            sum_dis += dis
        print('**********************************************************')
        print('EVAL: num={}, sum_dis={}, mean_dis={}'.format(num, sum_dis, 1.0*sum_dis/num))
        #to save check points
        args,auxs = mod.get_params()
        mx.rnn.save_rnn_checkpoint(stack, curr_path+'/rnn', epoch, net, args, auxs)



if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        if sys.argv[1] == 'gpu0':
            ctx = mx.gpu(0)
        elif sys.argv[1] == 'gpu1':
            ctx = mx.gpu(1)
    train()
