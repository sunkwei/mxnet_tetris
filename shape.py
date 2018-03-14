#!/bin/env python
#coding: utf-8

import random
#random.seed(0)

class ShapeFactory:
    def create(self, top_center):
        ''' top_center=(0, 7), 指的是创建的形状尽量使用 (0, 7) 作为其实位置
        '''
        n = random.randint(1, 7)    # 7 种基本形状
        # TEST:
        #n = 3 # 总生成方块, 用于训练．．
        # while n == 2:
        #     n = random.randint(1,7)
        class_name = 'Shape_{}'.format(n)
        shape = eval(class_name)(top_center)
        return shape


class Shape(object):
    def __init__(self):
        # poss 为占用的四个坐标  [(y,x), (y,x), (y,x), (y,x)]
        self.poss_ = []

    def d(self):
        assert(False)

    def __str__(self):
        return str(self.poss_)

    def poss(self):
        return self.poss_

    def set_poss(self, poss):
        self.poss_ = poss

    def left(self):
        # self.poss_ 左移之后的坐标
        dst = []
        for p in self.poss_:
            pl = (p[0], p[1]-1)
            dst.append(pl)
        return dst

    def right(self):
        dst = []
        for p in self.poss_:
            pr = (p[0], p[1]+1)
            dst.append(pr)
        return dst

    def down(self):
        dst = []
        for p in self.poss_:
            pd = (p[0]+1, p[1])
            dst.append(pd)
        return dst

    def mul(self, v, M):
        # 列向量乘以矩阵得到列向量
        result = []
        for r in M:     # M 的每行
            s = 0
            for i,vr in enumerate(v):
                s += vr * r[i]
            result.append(s)
        return result

    def rotate(self):
        ''' 其实这种单方向旋转的形状, 可以将a,c,d标记为从b出发的向量, 顺时针旋转就是乘以 [0, 1]
                                                                             [-1, 0]
            比如 a [-1] x [0, 1] = [0]  就是说 a 旋转后, 相对 b 的位置变为 [0]
                   [0 ]   [-1,0]   [1]                                [1]
        '''
        # 使用 b 作为圆心
        M = ((0,1),(-1,0))  # 顺时针旋转90度变换矩阵
        b = self.poss_[1]
        vs = [(y-b[0],x-b[1]) for y,x in self.poss_]    # a,b,c,d 基于 b 的向量
        rvs = [self.mul(v, M) for v in vs]      # 旋转后基于 b 的向量
        return [(y+b[0],x+b[1]) for y,x in rvs] # 加上 b 的偏移


class Shape_1(Shape):
    '''  a b          d
           c d      b c
                    a
    '''
    def __init__(self, c):
        super(Shape_1, self).__init__()
        mode = random.randint(0, 1)
        if mode == 0:
            self.poss_ = [(1,c-1), (1,c), (2,c), (2,c+1)]
        else:
            self.poss_ = [(2,c-1), (1,c-1), (1,c), (0,c)]

    def d(self):
        return 1

    def mode(self):
        p0 = self.poss_[0]
        p1 = self.poss_[1]
        if p0[0] == p1[0]:
            return 0
        else:
            return 1

    def rotate(self):
        ps = []
        a,b,c,d = self.poss_
        if self.mode() == 0:
            # 第三个点为圆心, 逆时针旋转
            ps.append((a[0]+2, a[1]))
            ps.append((b[0]+1, b[1]-1))
            ps.append(c)
            ps.append((d[0]-1, d[1]-1))
        else:
            # 第三个点为圆心, 顺时针旋转
            ps.append((a[0]-2, a[1]))
            ps.append((b[0]-1, b[1]+1))
            ps.append(c)
            ps.append((d[0]+1, d[1]+1))
        return ps


class Shape_2(Shape):
    '''     b a            a
          d c              b c
                             d
    '''
    def __init__(self, c):
        super(Shape_2, self).__init__()
        mode = random.randint(0, 1)
        if mode == 0:
            self.poss_ = [(1,c+1), (1,c), (2,c), (2,c-1)]
        else:
            self.poss_ = [(0, c), (1,c), (1,c+1), (2,c+1)]

    def d(self):
        return 2

    def mode(self):
        a = self.poss_[0]
        b = self.poss_[1]
        if a[0] == b[0]:
            return 0
        else:
            return 1

    def rotate(self):
        ps = []
        a,b,c,d = self.poss_
        if self.mode() == 0:
            # 使用第二个点逆时针旋转
            ps.append((a[0]-1, a[1]-1))
            ps.append(b)
            ps.append((c[0]-1,c[1]+1))
            ps.append((d[0], d[1]+2))
        else:
            # 使用第二个点, 顺时针旋转
            ps.append((a[0]+1,a[1]+1))
            ps.append(b)
            ps.append((c[0]+1,c[1]-1))
            ps.append((d[0],d[1]-2))
        return ps


class Shape_3(Shape):
    '''   a b
          d c
    '''
    def __init__(self, c):
        super(Shape_3, self).__init__()
        self.poss_ = [(0,c), (0,c+1), (1,c+1), (1,c)]

    def d(self):
        return 3

    def rotate(self):
        return self.poss_


class Shape_4(Shape):
    '''         d c          d       a        c b a 
                  b      a b c       b        d      
                  a                  c d       
    '''
    def __init__(self, c):
        super(Shape_4, self).__init__()
        mode = random.randint(0, 3)
        self.poss_ = [(2,c), (1,c), (0,c), (0,c-1)]
        for i in range(mode):
            self.poss_ = self.rotate()

    def d(self):
        return 4

    
class Shape_5(Shape):
    '''         a       d          c d       a b c  
                b       c b a      b             d  
              d c                  a                
    '''
    def __init__(self, c):
        super(Shape_5, self).__init__()
        mode = random.randint(0,3)
        self.poss_ = [(0,c),(1,c),(2,c),(2,c-1)]
        for i in range(mode):
            self.poss_ = self.rotate()

    def d(self):
        return 5


class Shape_6(Shape):
    '''                 a
            a b c d     b
                        c
                        d
    '''
    def __init__(self, c):
        super(Shape_6, self).__init__()
        mode = random.randint(0,1)
        if mode == 0:
            self.poss_ = [(1,c-1), (1,c), (1,c+1), (1,c+2)]
        else:
            self.poss_ = [(0,c),(1,c),(2,c),(3,c)]
        
    def d(self):
        return 6

    def mode(self):
        a,b = self.poss_[:2]
        return 0 if a[0] == b[0] else 1

    def rotate(self):
        # 使用 b 旋转
        ps = []
        a,b,c,d = self.poss_
        if self.mode() == 0:
            ps.append((a[0]-1,a[1]+1))
            ps.append(b)
            ps.append((c[0]+1,c[1]-1))
            ps.append((d[0]+2,d[1]-2))
        else:
            ps.append((a[0]+1,a[1]-1))
            ps.append(b)
            ps.append((c[0]-1,c[1]+1))
            ps.append((d[0]-2,d[1]+2))
        return ps


class Shape_7(Shape):
    '''           a      d      c                           
                d b    c b a    b d      a b c  
                  c             a          d

            其实这种单方向旋转的形状, 可以将a,c,d标记为从b出发的向量, 顺时针旋转就是乘以 [0, 1]
                                                                                   [-1, 0]
            比如 a [-1] x [0, 1] = [0]  就是说 a 旋转后, 相对 b 的位置变为 [0]
                   [0 ]   [-1,0]   [1]                                   [1]
    '''
    def __init__(self, c):
        super(Shape_7, self).__init__()
        mode = random.randint(0,3)
        self.poss_ = [(0,c),(1,c),(2,c),(1,c-1)]
        for i in range(mode):
            self.poss_ = self.rotate()

    def d(self):
        return 7



if __name__ == '__main__':
    s7 = Shape_7(5)
    print(s7)
    p = s7.rotate()
    print(p)
