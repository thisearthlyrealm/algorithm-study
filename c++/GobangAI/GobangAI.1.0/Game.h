//
// Created by Administrator on 2026/6/24.
//

#ifndef GOBANGAI_GAME_H
#define GOBANGAI_GAME_H

#include "ChessBoard.h"
#include "HumanPlayer.h"
#include "LocalAIPlayer.h"
#include "RecordManager.h"
#include <string>
#include <vector>
using namespace std;

class Game {
private:
    ChessBoard chessBoard;
    HumanPlayer blackPlayer;
    HumanPlayer whiteHumanPlayer;
    LocalAIPlayer whiteAIPlayer;
    RecordManager recordManager;

    static void showMenu();
    void showRules();
    void setAIDifficulty();
    void play(Player* black,Player* white,const string& mode);
    void changePlayer(Player*& currentPlayer,Player* black,Player* white);
    void undoMove(vector<pair<int,int>>& history,vector<string>& moveRecords,Player*& currentPlayer,Player* black,Player* white);
public:
    Game();
    void run();
};


#endif //GOBANGAI_GAME_H
