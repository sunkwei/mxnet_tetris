#!/bin/env python
#coding: utf-8

import mxnet as mx
import numpy as np
import sqlite3 as sq
import os.path as osp
from db import DB

ROWS = 20
COLS = 14
BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.01
CLASSES = 4 # XXX: 将0,4,5 当中一类

class Batch:
    def __init__(self, data, label):
        self.data_ = data
        self.label_ = label

    @property
    def data(self):
        return [mx.nd.array(self.data_)]

    @property
    def label(self):
        return [mx.nd.array(self.label_)]


class DataSource:
    def __init__(self, source='train'):
        self.db_ = DB(table=source)
        
    def next_batch(self, batch_size=128):
        rs = self.db_.get_random_rs(batch_size, flags=1)
        data = []
        label = []
        for r in rs:
            # r[0]=rows, r[1]=cols, r[2]=str data, r[3]=label
            # 将 r[2], r[3] 转换为 mx.nd 数据
            # r[2] 字符串是 ',' 分割
            dd = r[2].split(',')[:-1]   # 最后多了一个 ','
            d = [ 0 if int(x)==0 else 1 for x in dd ]   # 将所有非 0 数据修改为 1
            assert(len(d) == r[0] * r[1])
            assert(r[0] == ROWS and r[1] == COLS)
            l = r[3]
            # XXX: 将 l=0,3,4 视为一类, 都作为 0 吧, 这一类对应着 "已经不再需要调整" 了
            if l == 0 or l == 4 or l == 5:
                l = 0
            data.append(np.array(d).reshape((1,1,r[0],r[1])))   # shape=(1,1,ROWS,COLS)
            label.append(np.array(l).reshape(1))
        data = np.vstack(tuple(data))
        label = np.vstack(tuple(label)).reshape(len(rs))
        return Batch(data, label)


def gen_net1():
    ''' 构造一个卷积网络吧
    '''
    data = mx.sym.Variable(name='data') # 对应 Batch 中的 "data", shape=[batch_size, 1, ROWS, COLS]
    label = mx.sym.Variable(name='label') # 对应 Batch 中的 "label", shape=[batch_size]
    net = mx.sym.Convolution(data=data, num_filter=64, pad=(1,1), kernel=(4,4), stride=(1,1))  # (19, 13)
    net = mx.sym.BatchNorm(data=net)
    net = mx.sym.Convolution(data=net, num_filter=128, kernel=(3,3), stride=(2,2))
    net = mx.sym.Convolution(data=net, num_filter=256, kernel=(1,1), stride=(2,2))
    net = mx.sym.BatchNorm(data=net)
    net = mx.sym.Activation(data=net, act_type='tanh')
    net = mx.sym.flatten(data=net)
    net = mx.sym.FullyConnected(data=net, num_hidden=128)
    net = mx.sym.Activation(data=net, act_type='relu')
    net = mx.sym.FullyConnected(data=net, num_hidden=32)
    net = mx.sym.Activation(data=net, act_type='relu')
    net = mx.sym.FullyConnected(data=net, num_hidden=CLASSES)     # 映射 4 个类别
    net = mx.sym.SoftmaxOutput(data=net, label=label)
    return net

def gen_net():
    # 照理说, 应该构造一个 rnn 网络, 因为从一个形状诞生到停止活动, 是一个序列
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    data = mx.sym.Convolution(data, num_filter=16, pad=(1,1), kernel=(4,4), stride=(1,1))
    data = mx.sym.BatchNorm(data)
    data = mx.sym.Activation(data, act_type='tanh')
    data = mx.sym.flatten(data)
    data = mx.sym.FullyConnected(data, num_hidden=64)
    data = mx.sym.Activation(data, act_type='tanh')
    data = mx.sym.FullyConnected(data, num_hidden=CLASSES)
    data = mx.sym.Activation(data, act_type='tanh')
    data = mx.sym.SoftmaxOutput(data, label=label)
    return data

def train(epoch=-1):
    mod = mx.mod.Module(gen_net(), data_names=('data',), label_names=('label',), context=mx.cpu(0))
    mod.bind(data_shapes=(('data', (BATCH_SIZE, 1, ROWS, COLS)),), 
            label_shapes=(('label', (BATCH_SIZE,)),), for_training=True)
    if epoch >= 0:
        sym,args,auxs = mx.model.load_checkpoint('mx', epoch)
        mod.set_params(args, auxs)
    else:
        mod.init_params(initializer=mx.init.Xavier())
#    mod.init_optimizer('sgd', optimizer_params=(('learning_rate', LR),('momentum', 0.9),('wd', 0.05)))
    mod.init_optimizer('adam', optimizer_params=(('learning_rate', LR),))
    ds_trian = DataSource(source='train')
    ds_val = DataSource(source='val')
    for i in range(EPOCHS):
        for x in range(100):
            # FIXME: 现在 db 的数据只是随机取出一个批次，照理说，应该根据记录数 ...
            batch = ds_trian.next_batch(batch_size=BATCH_SIZE)
            mod.forward_backward(batch)
            mod.update()
            if (x+1)%10 == 0:
                out = mod.get_outputs()[0].asnumpy()
                mids = np.argmax(out,axis=1)
                matched = np.where(mids == batch.label_)
                good = len(matched[0])
                print('TRAIN: #{}/#{}, acc: {}'.format(i, x, good*1.0/BATCH_SIZE))
            # 
        # 校验
        op_good = 0
        op_total = 1*BATCH_SIZE
        for x in range(1):  # 10个批次的校验
            batch = ds_val.next_batch(batch_size=BATCH_SIZE)
            mod.forward(batch, is_train=False)
            out = mod.get_outputs()[0].asnumpy()
            # out.shape = (batch_size, 5)
            mids = np.argmax(out,axis=1)
            matched = np.where(mids == batch.label_)
            op_good += len(matched[0])
        print('VAL: #{}, acc:{}'.format(i, op_good*1.0/op_total))
        mod.save_checkpoint('mx', i)


if __name__ == '__main__':
    train(epoch=-1)
