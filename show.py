#!/bin/env python
#coding: utf-8

import cv2
import numpy as np
import os.path as osp

curr_path = osp.dirname(osp.abspath(__file__))

class Show:
    def __init__(self, width=14, height=20):
        self.width_ = width
        self.height_ = height
        self.colors_ = [
            (0,0,0),        # 空白
            (0,255,255),
            (255,0,255),
            (0,0,255),
            (0,255,0),
            (200,128,128),
            (255,255,0),
            (255,255,255),
        ]
        self.dx_ = 30   # 每个小方块的大小
        self.dy_ = 30
        cv2.namedWindow('game')
        #self.canvas_ = np.zeros((800, 1200, 3), dtype=np.uint8)
        #self.canvas_[:,:,:] = 255    # 白底
        self.canvas_ = cv2.imread(osp.sep.join((curr_path, 'background.jpg')))
        self.canvas_ = cv2.resize(self.canvas_, (1200,800))
        self.xoff_ = 160
        self.yoff_ = 70
        x1,y1,x2,y2 = self.xoff_-2,self.yoff_-2,self.xoff_+self.dx_*self.width_+2,self.yoff_+self.dy_*self.height_+2
        self.canvas_[y1:y2,x1:x2,:] = (64,64,64)
        cv2.rectangle(self.canvas_, (x1,y1), (x2,y2), (128,128,128), 2)

    def wait_key(self, timeout):
        # 等待键盘
        cv2.imshow('game', self.canvas_)
        key = cv2.waitKey(timeout)
        return key

    def show(self, data):
        # data 为二维数组, 0 为空白, 1,2,..7 对应 7 种颜色
        assert(len(data) == self.height_)
        assert(len(data[0]) == self.width_)
        for r,row in enumerate(data):
            for c,cc in enumerate(row):
                # cc 为数据内容, r, c 为行列号
                x1,y1 = self.xoff_+c*self.dx_, self.yoff_+r*self.dy_
                x2,y2 = x1+self.dx_, y1+self.dy_
                cv2.rectangle(self.canvas_, (x1+1,y1+1), (x2-1,y2-1), self.colors_[cc], -1)

    def show_gameover(self):
        y = self.yoff_ + 200
        x = self.xoff_ + 50
        self.canvas_[y:y+70, x:x+350] = (220,220,220)
        cv2.putText(self.canvas_, "GAME OVER", (x+20,y+40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 3)

    def show_info(self, info):
        y = self.yoff_ + 100
        x = self.xoff_ + self.dx_ * self.width_ + 120
        self.canvas_[y:y+80,x:x+450] = (255,255,255)
        cv2.putText(self.canvas_, info, (x,y+40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,0), 2)


if __name__ == '__main__':
    import random
    w,h = 14,20
    s = Show(w,h)
    # 生成测试 data
    data = [[random.randint(0,7) for x in range(w)] for y in range(h)]
    s.show(data)
    s.wait_key(0)
