//
// Created by Administrator on 2026/6/26.
//

#include "RecordManager.h"
#include <iostream>
#include <fstream>
#include <ctime>
using namespace std;

void RecordManager::saveRecord(const string &mode,const string &result,const vector<string>& moves) {
    ofstream fout("../record.txt",ios::app);
    if (!fout) {
        cout<<"Failed to open record file."<<endl;
        return;
    }
    time_t now=time(nullptr);
    string timeStr=ctime(&now);
    if (!timeStr.empty() && timeStr.back()=='\n') {
        timeStr.pop_back();
    }

    fout<<"Time: "<<timeStr<<endl;
    fout<<"Mode: "<<mode<<endl;
    fout<<"Result: "<<result<<endl;
    fout<<"Moves: "<<endl;
    for (int i=0;i<moves.size();i++) {
        fout<<moves[i]<<endl;
    }
    fout<<"-----------------------------"<<endl;
    fout.close();
}
void RecordManager::showRecords() {
    ifstream fin("../record.txt");
    if (!fin) {
        cout<<"No records yet."<<endl;
        cout<<endl;
        return;
    }

    cout<<endl;
    cout<<"===== Game Records ====="<<endl;
    string line;
    bool empty=true;

    while (getline(fin,line)) {
        empty=false;
        cout<<line<<endl;
    }

    if (empty) {
        cout<<"No records yet."<<endl;
    }
    cout<<endl;
    fin.close();
}
