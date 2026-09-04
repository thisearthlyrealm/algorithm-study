#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int solution(vector<vector<int>>& triangle) {
    int n=triangle.size();
    vector<vector<int>> dp(n,vector<int>(n,0));
    dp[0][0]=triangle[0][0];
    for (int i=1; i<n; i++) {
        for (int j=0; j<i+1; j++) {
            if (j==0) {
                dp[i][j]=triangle[i][j]+dp[i-1][j];
            } else if (j==i) {
                dp[i][j]=triangle[i][j]+dp[i-1][i-1];
            } else {
                dp[i][j]=triangle[i][j]+min(dp[i-1][j-1],dp[i-1][j]);
            }
        }
    }
    return *min_element(dp[n-1].begin(),dp[n-1].end());
}
int main() {
    vector<vector<int>> triangle={{2},{3,4},{6,5,7},{4,1,8,3}};
    int ans=solution(triangle);
    cout<<ans<<endl;
    return 0;
}