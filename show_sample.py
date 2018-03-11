#!/bin/env python
#coding: utf-8

import numpy as np
import sys
import os.path as osp
from inference_rnn import _load_npz

fname = sys.argv[1]
imgs, keys = _load_npz(fname, padding=False)

def show_img(img, idx, key):
    ''' 显示 img 当前的内容
    '''
    imgx = img.reshape((20,14)).astype(dtype=np.int8)
    ll = [ r.tolist() for r in imgx ]
    if idx == 0:
        print('NEW SEQ...')
    print('------------ {} key={} ------------'.format(idx, key))
    for i,l in enumerate(ll):
        s = [ str(c) for c in l ]
        print('\t{:0>2d}   - '.format(i) + ''.join(s))

imgs = imgs.reshape(imgs.shape[1:])
keys = keys.reshape(keys.shape[1:])

print(imgs.shape)
print(keys.shape)

i = 0
for img,key in zip(imgs, keys):
    show_img(img, i, key)
    i += 1
