//
// Created by Administrator on 2026/6/25.
//

#include "HumanPlayer.h"
#include <iostream>
#include <limits>
using namespace std;

HumanPlayer::HumanPlayer(int piece, string name):Player(piece,name){}

pair<int,int> HumanPlayer::move(ChessBoard& chessboard) {
    int x,y;
    while (true) {
        cout<<name<<" move."<<endl;
        cout<<"Please input row and col: ";
        if (!(cin>>x>>y)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout<<"Input error, please input two integers."<<endl;
            cout<<endl;
            continue;
        }
        return {x,y};
    }
}