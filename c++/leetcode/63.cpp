#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int solution(vector<vector<int>>& obstacleGrid) {
    int m=obstacleGrid.size();
    int n=obstacleGrid[0].size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    dp[1][1]=1;
    for (int i=1; i<=m; i++) {
        for (int j=1; j<=n; j++) {
            if (i==1 && j==1) {
                continue;
            }
            if (obstacleGrid[i-1][j-1]==1) {
                dp[i][j]=0;
            }
            else {
                dp[i][j]=max(dp[i][j], dp[i-1][j]+dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}
int main() {
    vector<vector<int>> obstacleGrid={
        {0,0,0},
        {0,1,0},
        {0,0,0}
    };
    int ans=solution(obstacleGrid);
    cout<<ans<<endl;
    return 0;
}
