#!/bin/env python
#coding: utf-8


import mxnet as mx
import numpy as np
import os.path as osp
from train_rnn import RNN_HIDDEN_NUM, RNN_LAYERS_NUM, MAX_SEQ_LENGTH, OUT_NUM


curr_path = osp.dirname(osp.abspath(__file__))


class Batch:
    def __init__(self, data, last_state=None):
        self.data_ = data
        self.last_state_ = last_state

    @property
    def data(self):
        ''' data_shapes, init_state_N
        '''
        ret = [ mx.nd.array(self.data_) ]
        if self.last_state_:
            ret += [ s for s in self.last_state_ ]
        return ret


class Inference:
    ''' 使用 train_rnn.py 训练的模型进行预测
    '''
    def __init__(self, prefix=curr_path+'/rnn', epoch=1):
        self.prefix_ = prefix
        self.epoch_ = epoch
        self.hidden_num_ = RNN_HIDDEN_NUM
        self.layer_num_ = RNN_LAYERS_NUM
        net, stack = self.build_inference_net(self.hidden_num_, self.layer_num_)
        self.mod_ = self.build_mod(net, stack)
        self.reset()

    def build_inference_net(self, hidden_num, layer_num):
        data = mx.sym.var('data')  # 接收 image (1,1,20,14)
        init_states = [mx.sym.var('init_state_{}'.format(i), 
                shape=(1,self.hidden_num_)) for i in range(self.layer_num_) ]
        
        # conv1 = mx.sym.Convolution(data, name='conv1', kernel=(4,4), pad=(1,1), num_filter=16)
        # act1 = mx.sym.Activation(conv1, act_type='tanh')
        # conv2 = mx.sym.Convolution(act1, name='conv2', kernel=(4,4), stride=(2,2), num_filter=64)
        # act2 = mx.sym.Activation(conv2, act_type='relu')
        # conv3 = mx.sym.Convolution(act2, name='conv3', kernel=(3,3), stride=(2,2), num_filter=128)
        # act3 = mx.sym.Activation(conv3, act_type='relu')
        # data = mx.sym.reshape(act3, (0,1,-1))

        stack = mx.rnn.SequentialRNNCell()
        for i in range(layer_num):
            cell = mx.rnn.GRUCell(hidden_num, prefix='gru_{}_'.format(i))
            stack.add(cell)
        outputs, states = stack.unroll(1, data, begin_state=init_states)
        fc = mx.sym.FullyConnected(outputs[0], num_hidden=64, name='fc1')
        act = mx.sym.Activation(fc, act_type='relu')
        fc = mx.sym.FullyConnected(act, num_hidden=OUT_NUM, name='fc2') # 将使用 params 中的 fc_weight/fc_bias
        pred = mx.sym.softmax(fc, axis=1)
        outs = [pred]
        outs.extend([s for s in states ]) # states 将作为下一个输入的 init_state_N 输入
        return mx.sym.Group(outs), stack

    def build_mod(self, net, stack):
        ''' XXX: init_state_XX 通过 set_params 传递 ?
        '''
        data_names = ['data']  # image
        data_names += [ 'init_state_%d'%i for i in range(self.layer_num_) ] # init_state_N
        data_shapes = [('data', (1,1,20,14))] # image
        data_shapes += [('init_state_%d'%i,(1,self.hidden_num_)) for i in range(self.layer_num_)]
        # 
        mod = mx.mod.Module(net, data_names=data_names, label_names=None)
        mod.bind(data_shapes=data_shapes, for_training=False)
        _, self.args_, self.auxs_ = mx.rnn.load_rnn_checkpoint(stack, self.prefix_, self.epoch_)
        self.internals_ = net.get_internals()
        self.set_params(mod)
        return mod

    def set_params(self, mod):
        ''' FIXME: 应该有更好的方法吧 ??
            因为需要动态设置 init_state_XX, fc_weight, fc_bias, 每次都 force_init ?
        '''
        need_args = {}
        for name in self.internals_.list_arguments():
            if name in self.args_:
                need_args[name] = self.args_[name]
            else:
                print('set_params: {} NOT set'.format(name))
        mod.set_params(self.args_, self.auxs_)

    def reset(self):
        ''' 重置 init_state_XXX 变量，在出现新形状时调用
        '''
        self.step_ = 0
        self.last_state_ = [ mx.nd.array(np.zeros((1, self.hidden_num_))) for i in range(self.layer_num_) ]

    def show_img(self, img, idx):
        ''' 显示 img 当前的内容
        '''
        imgx = img.reshape((20,14)).astype(dtype=np.int8)
        ll = [ r.tolist() for r in imgx ]
        if idx == 0:
            print('NEW SEQ...')
        print('------------ {} ------------'.format(idx))
        for l in ll:
            s = [ str(c) for c in l ]
            print('\t'+''.join(s))

    def pred(self, img):
        ''' 输入 img 为 numpy array, shape = (20,14)
            输出为 [0,1,2,3,4,5] 中的一个

            img 需要二值化
        '''
        img = img.astype(np.float32)
        idx = img[:,:] > 0  # 二值化
        img[idx] = 1.0
        img -= 0.1
        self.show_img(img, self.step_)
        img = img.reshape((1,1,20,14))
        batch = Batch(img, self.last_state_)
        self.mod_.forward(batch)
        outs = self.mod_.get_outputs()
        self.last_state_ = outs[1:]     # last_state
        pred = outs[0].asnumpy()       # (batch, pred)
        print(pred)
        key = np.argmax(pred[0])
        self.step_ += 1
        print('=========== GOT key={} ==========='.format(key))
        return key


def _load_npz(fname, padding=True):
    ''' data['imgs'].shape = (n, rows, cols), n 为序列长度
        data['keys'].shape = (n,)  

        注意：imgs[0,::] 作为 init_stats, 
                keys[0] == 0，忽略
                目的是训练：imgs[1,::] 希望收到 keys[1]
                        imgs[2,::] 希望收到 keys[2]
                        ...
    '''
    data = np.load(fname)
    imgs = data['imgs'].astype(np.float32)
    keys = data['keys']
    n,rows,cols = imgs.shape
    if n >= MAX_SEQ_LENGTH:
        return None, None
    assert(len(keys) == n)
    assert(n < MAX_SEQ_LENGTH)
    # padding:
    if padding:
        zeros = np.zeros((1, rows, cols))
        for i in range(MAX_SEQ_LENGTH-n):
            imgs = np.concatenate((imgs,zeros))
            keys = np.concatenate((keys,[0]))
        return imgs.reshape((1,MAX_SEQ_LENGTH,rows,cols)), keys.reshape((1,MAX_SEQ_LENGTH)) # 1 为 batch size
    else:
        return imgs.reshape((1,n,rows,cols)), keys.reshape((1,n))


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        n = '0641042-1'
    else:
        n = sys.argv[1]

    predictor = Inference(epoch=4)
    imgs, keys = _load_npz(curr_path+'/rnn_test/{}.npz'.format(n))
    idx = imgs[:,:,:,:] > 0
    imgs[idx] = 1
    imgs = imgs.reshape((MAX_SEQ_LENGTH, 20, 14))
    imgs = np.split(imgs, MAX_SEQ_LENGTH)
    preds = []
    for i,img in enumerate(imgs):
        # if i == 2:
        #     continue
        k = predictor.pred(img)
        preds.append(k)
        if k == 1:
            break
    print('keys:', keys[0].tolist())
    print('pred:', preds)
