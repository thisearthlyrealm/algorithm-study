#include <iostream>
#include <vector>
using namespace std;
vector<vector<int>> ans;
void solution(vector<int> &s,int i) {
    if (i==s.size()) {
        ans.push_back(s);
        return;
    }
    for (int j=i;j<s.size();j++) {
        swap(s[i],s[j]);
        solution(s,i+1);
        swap(s[i],s[j]);
    }
}
int main() {
    vector<int> nums={1,2,3};
    solution(nums,0);
    for (auto &v:ans) {
        for (int x:v) cout<<x<<" ";
        cout<<endl;
    }
}
