#include <iostream>
#include <vector>
#include <queue>
using namespace std;
typedef long long ll;
const int MOD=998244353;
ll qpow(ll a,ll b) {
    ll ans=1;
    while (b) {
        if (b&1) ans=ans*a%MOD;
        a=a*a%MOD;
        b>>=1;
    }
    return ans;
}
ll inv(ll x) {
    return qpow(x,MOD-2);
}
ll solution(int n,const vector<int>& p,const vector<int>& fa,const vector<int>& w,vector<int>& cnt,vector<int>& sumq) {
    vector<int> rem=cnt;
    vector<ll> dp(n+1,0);
    vector<ll> acc(n+1,0);
    queue<int> q;
    for (int i=1;i<=n;i++) {
        if (cnt[i]==0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u=q.front();
        q.pop();
        if (u==1) continue;
        int father=fa[u];
        acc[father]=(acc[father]+1LL*w[u]*dp[u])%MOD;
        rem[father]--;
        if (rem[father]==0) {
            if (sumq[father]==0) {
                dp[father]=0;
            }else {
                ll val=(p[father]+sumq[father]+acc[father])%MOD;
                dp[father]=val*inv(sumq[father])%MOD;
            }
            q.push(father);
        }
    }
    return dp[1];
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    vector<int> p(n+1);
    for (int i=1;i<=n;i++) {
        cin>>p[i];
    }
    vector<int> fa(n+1,0);
    vector<int> w(n+1,0);
    vector<int> cnt(n+1,0);
    vector<int> sumq(n+1,0);
    for (int i=2;i<=n;i++) {
        int x,y;
        cin>>x>>y;
        fa[i]=x;
        w[i]=y;
        cnt[x]++;
        sumq[x]+=y;
    }
    const ll ans=solution(n,p,fa,w,cnt,sumq);
    cout<<ans<<"\n";
    return 0;
}