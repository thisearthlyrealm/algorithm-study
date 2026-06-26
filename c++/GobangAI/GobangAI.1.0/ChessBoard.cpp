//
// Created by Administrator on 2026/6/24.
//

#include "ChessBoard.h"
#include <iostream>
using namespace std;

ChessBoard::ChessBoard() {
    init();
}
void ChessBoard::init() {
    board.assign(BOARD_SIZE,vector<int>(BOARD_SIZE,0));
}
char ChessBoard::getSymbol(int value) const{
    if (value==0) return '.';
    if (value==1) return 'X';
    return 'O';
}
void ChessBoard::show() const {
    cout<<"   ";
    for (int i=0;i<BOARD_SIZE;i++) {
        if (i<10) cout<<i<<"  ";
        else cout<<i<<" ";
    }
    cout<<endl;
    for (int i=0;i<BOARD_SIZE;i++) {
        if (i<10) cout<<i<<"  ";
        else cout<<i<<" ";
        for (int j=0;j<BOARD_SIZE;j++) {
            cout<<getSymbol(board[i][j])<<"  ";
        }
        cout<<endl;
    }
}
bool ChessBoard::inBoard(int x,int y) const {
    return x>=0 && x<BOARD_SIZE && y>=0 && y<BOARD_SIZE;
}
bool ChessBoard::canMove(int x,int y) const {
    return inBoard(x,y) && board[x][y]==0;
}
bool ChessBoard::place(int x,int y,int player) {
    if (!canMove(x,y)) return false;
    board[x][y]=player;
    return true;
}
void ChessBoard::remove(int x,int y) {
    if (inBoard(x,y)) board[x][y]=0;
}
bool ChessBoard::checkWin(int x,int y,int player) const {
    int dx[4]={1,0,1,1};
    int dy[4]={0,1,1,-1};
    for (int i=0;i<4;i++) {
        int cnt=1;
        int nx=x+dx[i];
        int ny=y+dy[i];
        while (inBoard(nx,ny) && board[nx][ny]==player) {
            cnt++;
            nx+=dx[i];
            ny+=dy[i];
        }
        nx=x-dx[i];
        ny=y-dy[i];
        while (inBoard(nx,ny) && board[nx][ny]==player) {
            cnt++;
            nx-=dx[i];
            ny-=dy[i];
        }
        if (cnt>=5) return true;
    }
    return false;
}
bool ChessBoard::isFull() const {
    for (int i=0;i<BOARD_SIZE;i++) {
        for (int j=0;j<BOARD_SIZE;j++) {
            if (board[i][j]==0) return false;
        }
    }
    return true;
}
int ChessBoard::getCell(int x,int y) const {
    if (!inBoard(x,y)) return -1;
    return board[x][y];
}