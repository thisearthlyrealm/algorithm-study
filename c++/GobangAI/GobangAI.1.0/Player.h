//
// Created by Administrator on 2026/6/25.
//

#ifndef GOBANGAI_PLAYER_H
#define GOBANGAI_PLAYER_H

#include "ChessBoard.h"
#include <string>
#include <utility>
using namespace std;
class Player {
protected:
    int piece;
    string name;
public:
    Player(int piece, string name);
    virtual ~Player();

    [[nodiscard]] int getPiece() const;
    [[nodiscard]] string getName() const;

    virtual pair<int,int> move(ChessBoard& chessBoard)=0;
};


#endif //GOBANGAI_PLAYER_H
