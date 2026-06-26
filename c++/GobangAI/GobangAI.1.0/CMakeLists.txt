cmake_minimum_required(VERSION 4.2)
project(GobangAI)

set(CMAKE_CXX_STANDARD 17)

add_executable(GobangAI main.cpp
        ChessBoard.h
        ChessBoard.cpp
        Game.h
        Game.cpp
        Player.h
        Player.cpp
        HumanPlayer.h
        HumanPlayer.cpp
        LocalAIPlayer.h
        LocalAIPlayer.cpp
        RecordManager.h
        RecordManager.cpp)
