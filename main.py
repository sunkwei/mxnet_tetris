#!/bin/env python
#coding: utf-8

from game import Game


def main():
    game = Game(save_oper=True)
    while not game.gameover():
        game.step()
        if game.quit():
            return
    game.wait_quit()
    

if __name__ == '__main__':
    main()
