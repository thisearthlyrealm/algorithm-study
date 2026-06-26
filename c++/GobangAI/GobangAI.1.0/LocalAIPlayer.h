//
// Created by Administrator on 2026/6/25.
//

#ifndef GOBANGAI_LOCALAIPLAYER_H
#define GOBANGAI_LOCALAIPLAYER_H

#include "Player.h"

class LocalAIPlayer:public Player {
private:
    int searchDepth;

    int evaluatePoint(ChessBoard& chessBoard,int x,int y,int targetPiece);
    int getPatternScore(int count,int openEnds);

    int getThreatLevel(ChessBoard& chessBoard,int x,int y,int targetPiece);
    bool hasNeighbor(ChessBoard& chessBoard,int x,int y);
    bool isBoardEmpty(ChessBoard& chessBoard);
    vector<pair<int,int>> getCandidateMoves(ChessBoard& chessBoard);
    int evaluatePlayerBoard(ChessBoard& chessBoard,int targetPiece);
    int evaluateBoard(ChessBoard& chessBoard);
    int minimax(ChessBoard& chessBoard,int depth,int alpha,int beta,bool isMaxPlayer);
public:
    LocalAIPlayer(int piece,string name);
    void setSearchDepth(int depth);
    [[nodiscard]] int getSearchDepth() const;

    pair<int,int> move(ChessBoard& chessBoard) override;
};


#endif //GOBANGAI_LOCALAIPLAYER_H
