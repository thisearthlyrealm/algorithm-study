#include <iostream>
#include <vector>
#include <string>
using namespace std;

int solution(string& s) {
    int n=s.size();
    if (n==0 || s[0]=='0') {
        return 0;
    }
    vector<int> dp(n,0);
    dp[0]=1;
    for (int i=1;i<n;i++) {
        if (s[i]!='0') {
            dp[i]+=dp[i-1];
        }
        if (s[i-1]=='1' || (s[i-1]=='2' && s[i]<='6')) {
            dp[i]+=(i==1?1:dp[i-2]);
        }
    }
    return dp[n-1];
}
int main() {
    string s="226";
    int ans=solution(s);
    cout<<ans<<endl;
    return 0;
}
