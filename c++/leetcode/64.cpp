#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int solution(vector<vector<int>>& grid) {
    int m=grid.size();
    int n=grid[0].size();
    vector<vector<int>> dp(m,vector<int>(n,0));
    dp[0][0]=grid[0][0];
    for (int i=1;i<n;i++) {
        dp[0][i]=dp[0][i-1]+grid[0][i];
    }
    for (int i=1;i<m;i++) {
        dp[i][0]=dp[i-1][0]+grid[i][0];
    }
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (i==0 || j==0) {
                continue;
            }
            dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j];
        }
    }
    return dp[m-1][n-1];
}
int main() {
    vector<vector<int>> grid={
        {1,3,1},{1,5,1},{4,2,1}
    };
    int ans=solution(grid);
    cout<<ans<<endl;
    return 0;
}