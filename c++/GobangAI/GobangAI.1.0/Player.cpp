//
// Created by Administrator on 2026/6/25.
//

#include "Player.h"
using namespace std;
Player::Player(int piece,string name) {
    this->piece=piece;
    this->name=name;
}
Player::~Player() {}

int Player::getPiece() const {
    return piece;
}
string Player::getName() const {
    return name;
}