#include <iostream>
#include <vector>
using namespace std;
vector<vector<int>> solution(int numRows) {
    vector<vector<int>> ans={{1}};
    if (numRows==1) {
        return ans;
    }
    if (numRows==0) {
        return {};
    }
    for (int i=1; i<numRows; i++) {
        vector<int> pre=ans[i-1];
        vector<int> curr={1};
        for (int j=1; j<i; j++) {
            curr.push_back(pre[j-1]+pre[j]);
        }
        curr.push_back(1);
        ans.push_back(curr);
    }
    return ans;
}
int main() {
    int numRows;
    cin>>numRows;
    vector<vector<int>> ans=solution(numRows);
    cout<<"[";
    for (int i=0; i<numRows; i++) {
        if (i!=0) {
            cout<<",";
        }
        cout<<"[";
        for (int j=0; j<ans[i].size(); j++) {
            if (j!=0) {
                cout<<",";
            }
            cout<<ans[i][j];
        }
        cout<<"]";
    }
    cout<<"]";
}