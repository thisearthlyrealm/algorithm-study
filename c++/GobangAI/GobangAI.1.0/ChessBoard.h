//
// Created by Administrator on 2026/6/24.
//

#ifndef GOBANGAI_CHESSBOARD_H
#define GOBANGAI_CHESSBOARD_H

#include <vector>
constexpr int BOARD_SIZE = 15;
using namespace std;
class ChessBoard {
private:
    vector<vector<int>> board;
    [[nodiscard]]char getSymbol(int value) const;
public:
    ChessBoard();
    void init();
    void show() const;
    [[nodiscard]] bool inBoard(int x,int y) const;
    [[nodiscard]] bool canMove(int x,int y) const;
    bool place(int x,int y,int player);
    void remove(int x,int y);
    [[nodiscard]] bool checkWin(int x,int y,int player) const;
    [[nodiscard]] bool isFull() const;
    [[nodiscard]] int getCell(int x,int y) const;
};


#endif //GOBANGAI_CHESSBOARD_H
