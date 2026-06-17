#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
const int MOD=998544353;
int solution(int n,int k,string& s) {
    k=min(k,n);
    if (n==1) {
        int cnt=0;
        if (k>=0) cnt++;
        if (k>=1) cnt++;
        return cnt%MOD;
    }
    if (n==2) {
        int cnt=0;
        for (int i=0; i<=min(k,2);++i) {
            if (i==0) cnt=(cnt+1)%MOD;
            else if (i==1) cnt=(cnt+2)%MOD;
            else cnt=(cnt+1)%MOD;
        }
        return cnt%MOD;
    }
    vector<int> a(n+1);
    for (int i=1;i<=n;++i) {a[i]=s[i-1]-'0';}
    vector<vector<int>> dp(k+1,vector<int>(4,0));
    for (int b1=0;b1<=1;++b1) {
        for (int b2=0;b2<=1;++b2) {
            int cost=(a[1]!=b1)+(a[2]!=b2);
            if (cost>k) continue;
            int mask=(b1<<1)|b2;
            dp[cost][mask]=(dp[cost][mask]+1)%MOD;
        }
    }
    for (int i=3;i<=n;++i) {
        vector<vector<int>> ndp(k+1,vector<int>(4,0));
        for (int j=0;j<=k;++j) {
            for (int mask=0;mask<4;++mask) {
                int cur=dp[j][mask];
                if (cur==0) continue;
                for (int b=0;b<=1;++b) {
                    int nj=j+(b!=a[i]);
                    if (nj>k) continue;
                    if (mask==3 && b==0) continue;
                    int n_mask=((mask&1)<<1)|b;
                    ndp[nj][n_mask]=(ndp[nj][n_mask]+cur)%MOD;
                }
            }
        }
        dp=move(ndp);
    }
    int ans=0;
    for (int j=0;j<=k;++j) {
        for (int mask=0;mask<4;++mask) {
            ans=(ans+dp[j][mask])%MOD;
        }
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n,k;
    string s;
    cin>>n>>k;
    cin>>s;
    const int ans=solution(n,k,s);
    cout<<ans<<endl;
    return 0;
}