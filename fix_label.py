#!/bin/env python
#coding: utf-8


''' 如果 label 使用 0 开始的，则将 keys 前一，最后补一个 1
'''


from inference_rnn import _load_npz
import sys
import os
import os.path as osp
import numpy as np



path = sys.argv[1]
for fname in os.listdir(path):
    _, ext = osp.splitext(fname)
    if ext == '.npz':
        fname = path + '/' + fname
        imgs, keys = _load_npz(fname, padding=False)
        if imgs is None:
            continue

        imgs = imgs.reshape(imgs.shape[1:])
        keys = keys.reshape(keys.shape[1:])

        ks = keys.tolist()
        if ks[0] == 0:
            del ks[0]
            ks.append(1)    # 最后补１，保证长度
            keys = np.array(ks, dtype=np.int)

            np.savez_compressed(fname, imgs=imgs, keys=keys)

            print('{} converted!'.format(fname))
