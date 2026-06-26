//
// Created by Administrator on 2026/6/24.
//

#include "Game.h"
#include <iostream>
#include <limits>
using namespace std;

Game::Game():
blackPlayer(1,"Black X"),
whiteHumanPlayer(2,"White O"),
whiteAIPlayer(2,"Local AI O") {}

void Game::showMenu() {
    cout<<"===== Gobang AI System ====="<<endl;
    cout<<"1. Human VS Human"<<endl;
    cout<<"2. Human VS Local AI"<<endl;
    cout<<"3. Game Rules"<<endl;
    cout<<"4. Game Records"<<endl;
    cout<<"5. Set AI Difficulty"<<endl;
    cout<<"0. Exit"<<endl;
    cout<<"Please choose: ";
}

void Game::showRules() {
    cout<<endl;
    cout<<"===== Game Rules ====="<<endl;
    cout<<"1. The board size is 15 x 15."<<endl;
    cout<<"2. Black X moves first."<<endl;
    cout<<"3. Five connected pieces win."<<endl;
    cout<<"4. Input row and col to place a piece."<<endl;
    cout<<"5. Input -1 -1 during a game to quit."<<endl;
    cout<<"6. Input -2 -2 during a game to undo."<<endl;
    cout<<endl;
}

void Game::setAIDifficulty() {
    cout<<endl;
    cout<<"===== AI Difficulty ====="<<endl;
    cout<<"1. Easy search depth=1"<<endl;
    cout<<"2. Normal search depth=2"<<endl;
    cout<<"3. Hard search depth=3"<<endl;
    cout<<"Please choose difficulty: ";

    int choice;
    if (!(cin>>choice)) {
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(),'\n');
        cout<<"Input error, difficulty unchanged."<<endl;
        cout<<endl;
        return;
    }

    if (choice==1) {
        whiteAIPlayer.setSearchDepth(1);
        cout<<"AI difficulty set to Easy."<<endl;
    }else if (choice==2) {
        whiteAIPlayer.setSearchDepth(2);
        cout<<"AI difficulty set to Normal."<<endl;
    }else if (choice==3) {
        whiteAIPlayer.setSearchDepth(3);
        cout<<"AI difficulty set to Hard."<<endl;
    }else {
        cout<<"Invalid difficulty,unchanged."<<endl;
    }
    cout<<endl;
}

void Game::changePlayer(Player*& currentPlayer,Player* black,Player* white) {
    if (currentPlayer==black) currentPlayer=white;
    else currentPlayer=black;
}

void Game::undoMove(vector<pair<int,int>>& history,vector<string>&moveRecords,Player*& currentPlayer,Player* black,Player* white) {
    if (history.empty()) {
        cout<<"No move to undo."<<endl;
        cout<<endl;
        return;
    }
    int undoCount=1;
    if (white==&whiteAIPlayer && history.size()>=2) {
        undoCount=2;
    }
    for (int i=0;i<undoCount;i++) {
        pair<int,int> pos=history.back();
        history.pop_back();
        if (!moveRecords.empty()) {
            moveRecords.pop_back();
        }
        chessBoard.remove(pos.first,pos.second);
    }
    if (history.size()%2==0) currentPlayer=black;
    else currentPlayer=white;
    cout<<"Undo success."<<endl;
    cout<<endl;
}

void Game::play(Player* black,Player* white,const string& mode) {
    chessBoard.init();
    vector<pair<int,int>> history;
    vector<string> moveRecords;
    Player* currentPlayer=black;

    while (true) {
        chessBoard.show();
        cout<<endl;
        pair<int,int> pos=currentPlayer->move(chessBoard);
        int x=pos.first,y=pos.second;

        if (x==-1 && y==-1) {
            cout<<"Game quit."<<endl;
            break;
        }

        if (x==-2 && y==-2) {
            undoMove(history,moveRecords,currentPlayer,black,white);
            continue;
        }

        if (!chessBoard.place(x,y,currentPlayer->getPiece())) {
            cout<<"Invalid move, please try again."<<endl;
            cout<<endl;
            continue;
        }

        history.emplace_back(x,y);
        string moveInfo="Step "+to_string(history.size())+": "+currentPlayer->getName()+" -> ("+to_string(x)+","+to_string(y)+")";
        moveRecords.push_back(moveInfo);

        if (chessBoard.checkWin(x,y,currentPlayer->getPiece())) {
            chessBoard.show();
            cout<<endl;
            string result=currentPlayer->getName()+" wins";
            cout<<result<<"!"<<endl;

            recordManager.saveRecord(mode,result,moveRecords);
            break;
        }

        if (chessBoard.isFull()) {
            chessBoard.show();
            cout<<endl;
            string result="Draw game";
            cout<<result<<"."<<endl;

            recordManager.saveRecord(mode,result,moveRecords);
            break;
        }

        changePlayer(currentPlayer,black,white);
        cout<<endl;
    }
}

void Game::run() {
    while(true) {
        showMenu();

        int choice;
        if(!(cin>>choice)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(),'\n');
            cout<<"Input error, please input a number."<<endl;
            cout<<endl;
            continue;
        }

        if(choice==1) {
            cout<<endl;
            cout<<"Mode: Human VS Human"<<endl;
            play(&blackPlayer,&whiteHumanPlayer,"Human VS Human");
        }else if(choice==2) {
            cout<<endl;
            cout<<"Mode: Human VS Local AI"<<endl;
            cout<<"Current AI search depth: "<<whiteAIPlayer.getSearchDepth()<<endl;
            play(&blackPlayer,&whiteAIPlayer,"Human VS Local AI");
        }else if(choice==3) {
            showRules();
        }else if (choice==4) {
            recordManager.showRecords();
        }else if (choice==5) {
            setAIDifficulty();
        }else if(choice==0) {
            cout<<"System exit."<<endl;
            break;
        }else {
            cout<<"Invalid choice, please try again."<<endl;
            cout<<endl;
        }
    }
}


