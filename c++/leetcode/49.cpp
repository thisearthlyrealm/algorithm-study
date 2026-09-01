#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
using namespace std;

vector<vector<string>> solution(vector<string>& strs) {
    unordered_map<string,vector<string>> m;
    for (string& s: strs) {
        string sort_m=s;
        sort(sort_m.begin(), sort_m.end());
        m[sort_m].push_back(s);
    }
    vector<vector<string>> ans;
    for (auto& kv: m) {
        ans.push_back(kv.second);
    }
    return ans;
}
int main() {
    vector<string> strs={"eat", "tea", "tan", "ate", "nat", "bat"};
    vector<vector<string>> ans=solution(strs);
    cout<<"[";
    for (int g=0; g<ans.size(); g++) {
        if (g>0) cout<<",";
        cout<<"[";
        for (int i=0; i<ans[g].size(); i++) {
            if (i>0) cout<<",";
            cout<<"\""<<ans[g][i]<<"\"";
        }
        cout<<"]";
    }
    cout<<"]";
}
