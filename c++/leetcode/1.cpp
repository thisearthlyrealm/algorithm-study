#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;
vector<int> solution(vector<int>& nums,int target) {
    unordered_map<int,int> mp;
    for (int i=0;i<nums.size();++i) {
        int complement=target-nums[i];
        if (mp.count(complement)) {
            return {mp[complement],i};
        }
        mp[nums[i]]=i;
    }
    return {};
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n,target;
    cin>>n>>target;
    vector<int> nums(n);
    for (int i=0;i<n;++i) cin>>nums[i];
    vector<int> ans=solution(nums,target);
    cout<<ans[0]<<" "<<ans[1]<<endl;
    return 0;
}