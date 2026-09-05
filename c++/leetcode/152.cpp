#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int solution(vector<int>& nums) {
    int n=nums.size();
    if (n==0) return 0;
    vector<int> max_dp(n,0);
    vector<int> min_dp(n,0);
    max_dp[0]=min_dp[0]=nums[0];
    for(int i=1;i<n;i++) {
        max_dp[i]=max({max_dp[i-1]*nums[i],min_dp[i-1]*nums[i],nums[i]});
        min_dp[i]=min({min_dp[i-1]*nums[i],max_dp[i-1]*nums[i],nums[i]});
    }
    return *max_element(max_dp.begin(),max_dp.end());
}
int main() {
    vector<int> nums={2,3,-2,4};
    int ans=solution(nums);
    cout<<ans<<endl;
    return 0;
}