//
// Created by Administrator on 2026/6/26.
//

#ifndef GOBANGAI_RECORDMANAGER_H
#define GOBANGAI_RECORDMANAGER_H

#include <string>
#include <vector>
using namespace std;

class RecordManager {
public:
    void saveRecord(const string& mode,const string& result,const vector<string>& moves);
    void showRecords();
};


#endif //GOBANGAI_RECORDMANAGER_H
