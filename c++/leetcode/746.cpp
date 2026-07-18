#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
int solution(vector<int>& cost) {
    int n=cost.size();
    vector<int> dp(n+3,INT_MAX);
    dp[0]=dp[1]=0;
    for (int i=0;i<n;i++) {
        dp[i+1]=min(dp[i+1],dp[i]+cost[i]);
        dp[i+2]=min(dp[i+2],dp[i]+cost[i]);
    }
    return dp[n];
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<int> cost={10,15,20};
    const int ans=solution(cost);
    cout<<ans<<endl;
    return 0;
}