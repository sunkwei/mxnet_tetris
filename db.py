#!/bin/env python
#coding: utf-8


import sqlite3
import os.path as osp


curr_path = osp.dirname(osp.abspath(__file__))


class DB:
    ''' 将每次按键时, 对应的 data 保存下来, 用于制作训练样本

        数据库格式:
            img: TEXT 使用 ',' 分割的字符串, 个数为 20*14
            lab: int 一个字符对应按键, 应该有 UP, LEFT, DOWN, RIGHT, SP 等操作
            flags: 1 训练数据，2 校验数据
    '''

    def __init__(self, fname=osp.sep.join((curr_path, 'samples.db')), table='train'):
        self.conn_ = sqlite3.connect(fname)
        self.table_name_ = table
        s = 'create table if not exists {} (row INTEGER, col INTEGER, img TEXT, lab INTEGER, flags INTEGER)'.format(table)
        self.conn_.execute(s)
        self.cnt_ = 0

    def save(self, row, col, data, lab, flags=-1):
        s = 'insert into {} values (?,?,?,?,?)'.format(self.table_name_)
        m = [row, col, data, lab, flags]
        self.conn_.execute(s, m)
        self.cnt_ += 1
        if self.cnt_ // 10 == 0:
            self.conn_.commit()

    def __del__(self):
        self.conn_.commit()
        self.conn_.close()

    def get_random_rs(self, n, flags=1):
        ''' 随机返回 n 条数据
        '''
        s = 'select * from {} where flags={} order by random() limit {}'.format(self.table_name_,flags,n)
        cur = self.conn_.cursor()
        cur.execute(s)
        rs = cur.fetchall()
        assert(len(rs) == n)
        return rs


if __name__ == '__main__':
    db = DB(table='val')
    db.save(3,3,'0,1,2,3,4,5,6,7,8,9,10', -1)
