#include <iostream>
#include <vector>
#include <unordered_set>
using namespace std;

vector<vector<int>> ans;
void solution(vector<int>& nums,int i) {
    if (i==nums.size()-1) {
        ans.push_back(nums);
        return;
    }
    unordered_set<int> used;
    for (int j=i;j<nums.size();j++) {
        if (used.find(nums[j])==used.end()) {
            used.insert(nums[j]);
            swap(nums[i],nums[j]);
            solution(nums,i+1);
            swap(nums[i],nums[j]);
        }
    }
}
int main() {
    vector<int> nums={1,1,2};
    solution(nums,0);
    for (auto &v:ans) {
        for (int x:v) cout<<x<<" ";
        cout<<endl;
    }
}