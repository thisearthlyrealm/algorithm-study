//
// Created by Administrator on 2026/6/25.
//

#ifndef GOBANGAI_HUMANPLAYER_H
#define GOBANGAI_HUMANPLAYER_H

#include "Player.h"
class HumanPlayer:public Player {
public:
    HumanPlayer(int piece,string name);
    pair<int,int> move(ChessBoard& chessboard) override;
};


#endif //GOBANGAI_HUMANPLAYER_H
