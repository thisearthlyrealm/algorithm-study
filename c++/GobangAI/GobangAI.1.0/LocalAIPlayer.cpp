//
// Created by Administrator on 2026/6/25.
//

#include "LocalAIPlayer.h"
#include <iostream>
#include <utility>
#include <algorithm>
using namespace std;

LocalAIPlayer::LocalAIPlayer(int piece,string name):Player(piece,name) {
    searchDepth=2;
}

void LocalAIPlayer::setSearchDepth(int depth) {
    if (depth<1) depth=1;
    if (depth>3) depth=3;
    searchDepth=depth;
}

int LocalAIPlayer::getSearchDepth() const {
    return searchDepth;
}

int LocalAIPlayer::getPatternScore(int count,int openEnds) {
    if (count>=5) return 1000000;
    if (count==4 && openEnds==2) return 100000;
    if (count==4 && openEnds==1) return 50000;
    if (count==3 && openEnds==2) return 10000;
    if (count==3 && openEnds==1) return 5000;
    if (count==2 && openEnds==2) return 1000;
    if (count==2 && openEnds==1) return 500;
    if (count==1 && openEnds==2) return 50;
    return 10;
}
bool LocalAIPlayer::isBoardEmpty(ChessBoard& chessBoard) {
    for (int i=0;i<BOARD_SIZE;i++) {
        for (int j=0;j<BOARD_SIZE;j++) {
            if (chessBoard.getCell(i,j)!=0) return false;
        }
    }
    return true;
}
bool LocalAIPlayer::hasNeighbor(ChessBoard& chessBoard,int x,int y) {
    for (int i=-2;i<=2;i++) {
        for (int j=-2;j<=2;j++) {
            if (i==0 && j==0) continue;
            int nx=x+i;
            int ny=y+j;
            if (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)!=0) {
                return true;
            };
        }
    }
    return false;
}
vector<pair<int,int>> LocalAIPlayer::getCandidateMoves(ChessBoard& chessBoard) {
    vector<pair<int,int>> candidates;
    for (int i=0;i<BOARD_SIZE;i++) {
        for (int j=0;j<BOARD_SIZE;j++) {
            if (!chessBoard.canMove(i,j)) continue;
            if (!hasNeighbor(chessBoard,i,j)) continue;
            candidates.emplace_back(i,j);
        }
    }
    if (candidates.empty()) {
        candidates.emplace_back(BOARD_SIZE/2,BOARD_SIZE/2);
    }
    return candidates;
}
int LocalAIPlayer::getThreatLevel(ChessBoard& chessBoard,int x,int y,int targetPiece) {
    int dx[4]={1,0,1,1};
    int dy[4]={0,1,1,-1};
    int maxLevel=0;
    for (int i=0;i<4;i++) {
        int count=1;
        int openEnds=0;

        int nx=x+dx[i];
        int ny=y+dy[i];
        while (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==targetPiece) {
            count++;
            nx+=dx[i];
            ny+=dy[i];
        }
        if (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==0) {
            openEnds++;
        }

        nx=x-dx[i];
        ny=y-dy[i];
        while (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==targetPiece) {
            count++;
            nx-=dx[i];
            ny-=dy[i];
        }
        if (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==0) {
            openEnds++;
        }

        int level=0;
        if (count>=5) level=6;
        else if (count==4 && openEnds==2) level=5;
        else if (count==4 && openEnds==1) level=4;
        else if (count==3 && openEnds==2) level=3;
        else if (count==3 && openEnds==1) level=2;
        else if (count==2 && openEnds==2) level=1;
        maxLevel=max(maxLevel,level);
    }
    return maxLevel;
}
int LocalAIPlayer::evaluatePoint(ChessBoard& chessBoard,int x,int y,int targetPiece) {
    int dx[4]={1,0,1,1};
    int dy[4]={0,1,1,-1};
    int totalScore=0;
    for (int i=0;i<4;i++) {
        int count=1;
        int openEnds=0;

        int nx=x+dx[i];
        int ny=y+dy[i];
        while (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==targetPiece) {
            count++;
            nx+=dx[i];
            ny+=dy[i];
        }
        if (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==0) {
            openEnds++;
        }

        nx=x-dx[i];
        ny=y-dy[i];
        while (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==targetPiece) {
            count++;
            nx-=dx[i];
            ny-=dy[i];
        }
        if (chessBoard.inBoard(nx,ny) && chessBoard.getCell(nx,ny)==0) {
            openEnds++;
        }

        totalScore+=getPatternScore(count,openEnds);
    }
    return totalScore;
}

int LocalAIPlayer::evaluatePlayerBoard(ChessBoard &chessBoard, int targetPiece) {
    int totalScore=0;
    for (int i=0;i<BOARD_SIZE;i++) {
        for (int j=0;j<BOARD_SIZE;j++) {
            if (chessBoard.getCell(i,j)==targetPiece) {
                totalScore+=evaluatePoint(chessBoard,i,j,targetPiece);
            }
        }
    }
    return totalScore;
}

int LocalAIPlayer::evaluateBoard(ChessBoard &chessBoard) {
    int opponent=(piece==1?2:1);
    int aiScore=evaluatePlayerBoard(chessBoard,piece);
    int opponentScore=evaluatePlayerBoard(chessBoard,opponent);
    return aiScore-opponentScore;
}

int LocalAIPlayer::minimax(ChessBoard &chessBoard,int depth,int alpha,int beta,bool isMaxPlayer) {
    if (depth==0 || chessBoard.isFull()) {
        return evaluateBoard(chessBoard);
    }
    vector<pair<int,int>> candidates=getCandidateMoves(chessBoard);
    int opponent=(piece==1?2:1);
    if (isMaxPlayer) {
        int bestScore=-100000000;
        for (auto pos:candidates) {
            int x=pos.first;
            int y=pos.second;
            chessBoard.place(x,y,piece);
            int score;
            if (chessBoard.checkWin(x,y,piece)) {
                score=100000000+depth;
            }else {
                score=minimax(chessBoard,depth-1,alpha,beta,false);
            }
            chessBoard.remove(x,y);
            bestScore=max(bestScore,score);
            alpha=max(alpha,bestScore);
            if (alpha>=beta) break;
        }
        return bestScore;
    }else {
        int bestScore=100000000;
        for (auto pos:candidates) {
            int x=pos.first;
            int y=pos.second;
            chessBoard.place(x,y,opponent);
            int score;
            if (chessBoard.checkWin(x,y,opponent)) {
                score=-100000000-depth;
            }else {
                score=minimax(chessBoard,depth-1,alpha,beta,true);
            }
            chessBoard.remove(x,y);
            bestScore=min(bestScore,score);
            beta=min(beta,bestScore);
            if (alpha>=beta) break;
        }
        return bestScore;
    }
}

pair<int, int> LocalAIPlayer::move(ChessBoard& chessBoard) {
    cout<<name<<" is thinking..."<<endl;
    int opponent=(piece==1?2:1);

    if (isBoardEmpty(chessBoard)) {
        cout<<name<<" move: "<<BOARD_SIZE/2<<" "<<BOARD_SIZE/2<<endl;
        return {BOARD_SIZE/2,BOARD_SIZE/2};
    }
    vector<pair<int,int>> candidates=getCandidateMoves(chessBoard);
    cout<<"Board score: "<<evaluateBoard(chessBoard)<<endl;
    for(auto pos:candidates) {
        int i=pos.first;
        int j=pos.second;

        chessBoard.place(i,j,piece);
        bool win=chessBoard.checkWin(i,j,piece);
        chessBoard.remove(i,j);
        if(win) {
            cout<<name<<" win move: "<<i<<" "<<j<<endl;
            return {i,j};
        }
    }

    for(auto pos:candidates) {
        int i=pos.first;
        int j=pos.second;

        chessBoard.place(i,j,opponent);
        bool opponentWin=chessBoard.checkWin(i,j,opponent);
        chessBoard.remove(i,j);

        if(opponentWin) {
            cout<<name<<" block move: "<<i<<" "<<j<<endl;
            return {i,j};
        }
    }

    int bestX=-1;
    int bestY=-1;
    int bestScore=-100000000;

    for(auto pos:candidates) {
        int x=pos.first;
        int y=pos.second;

        chessBoard.place(x,y,piece);

        int score;
        if(chessBoard.checkWin(x,y,piece)) {
            score=100000000;
        }else {
            score=minimax(chessBoard,searchDepth-1,-100000000,100000000,false);
        }

        chessBoard.remove(x,y);

        if(score>bestScore) {
            bestScore=score;
            bestX=x;
            bestY=y;
        }
    }

    if(bestX==-1||bestY==-1) {
        bestX=BOARD_SIZE/2;
        bestY=BOARD_SIZE/2;
    }

    cout<<name<<" minimax move: "<<bestX<<" "<<bestY<<" score: "<<bestScore<<endl;
    return {bestX,bestY};
}