#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
vector<vector<int>> solution(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    int n=nums.size();
    vector<vector<int>> ans;
    vector<int> path;

    auto dfs=[&](auto&& self, int i)->void {
        if (i==n) {
            ans.push_back(path);
            return;
        }
        int x=nums[i];
        path.push_back(x);
        self(self, i+1);
        path.pop_back();  // 恢复现场
        i++;
        while (i<n && nums[i]==x) {
            i++;
        }
        self(self, i);
    };
    dfs(dfs, 0);
    return ans;
}
int main() {
    vector<int> nums={1,2,2};
    vector<vector<int>> ans=solution(nums);
    for (auto& v: ans) {
        for (int x: v) {
            cout<<x<<" ";
        }
        cout<<endl;
    }
    return 0;
}
