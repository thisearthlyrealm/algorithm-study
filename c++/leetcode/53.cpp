#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int solution_1(vector<int>& nums) {
    int pre=nums[0], ans=nums[0];
    for (int i=1;i<nums.size();i++) {
        pre=max(nums[i],pre+nums[i]);
        ans=max(ans,pre);
    }
    return ans;
}
int solution_2(vector<int>& nums) {
    int n=nums.size();
    vector<int> dp(n,0);
    dp[0]=nums[0];
    for (int i=1;i<n;i++) {
        dp[i]=max(nums[i],dp[i-1]+nums[i]);
    }
    return *max_element(dp.begin(),dp.end());
}
int main() {
    vector<int> nums={-2,1,-3,4,-1,2,1,-5,4};
    int ans_1=solution_1(nums);
    int ans_2=solution_2(nums);
    cout<<ans_1<<","<<ans_2<<endl;
}