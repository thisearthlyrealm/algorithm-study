#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
int solution(int m,int n,vector<vector<int>>& waitCost) {
    vector<vector<int>> dp(m+1,vector<int>(n+1,INT_MAX));
    dp[0][1]=-waitCost[0][0];
    for (int i=1;i<=m;i++) {
        for (int j=1;j<=n;j++) {
            dp[i][j]=min(dp[i-1][j],dp[i][j-1])+i*j+waitCost[i-1][j-1];
        }
    }
    return dp[m][n]-waitCost[m-1][n-1];
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int m=2,n=3;
    vector<vector<int>> waitCost={{6,1,4},{3,2,5}};
    const int ans=solution(m,n,waitCost);
    cout<<ans<<endl;
    return 0;
}