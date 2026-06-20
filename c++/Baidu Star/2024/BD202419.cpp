#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
typedef long long ll;
const int MOD1=1e9+7;
const int MOD2=1e9+9;
const int BASE=911382323;
int n,q;
string s;
vector<ll> pow1,pow2;
vector<vector<ll>> h1,h2;
pair<ll,ll> get_hash(int c,int l,int r) {
    ll x=(h1[c][r]-h1[c][l-1]*pow1[r-l+1]%MOD1+MOD1)%MOD1;
    ll y=(h2[c][r]-h2[c][l-1]*pow2[r-l+1]%MOD2+MOD2)%MOD2;
    return {x,y};
}
void init() {
    pow1.assign(n+1,1);
    pow2.assign(n+1,1);
    for (int i=1;i<=n;++i) {
        pow1[i]=pow1[i-1]*BASE%MOD1;
        pow2[i]=pow2[i-1]*BASE%MOD2;
    }
    h1.assign(26,vector<ll>(n+1,0));
    h2.assign(26,vector<ll>(n+1,0));
    for (int c=0;c<26;++c) {
        for (int i=1;i<=n;++i) {
            int val=(s[i-1]-'a'==c);
            h1[c][i]=(h1[c][i-1]*BASE+val)%MOD1;
            h2[c][i]=(h2[c][i-1]*BASE+val)%MOD2;
        }
    }
}

void solution(int l1, int r1, int l2, int r2) {
    string ans;
    for (int c=0;c<26;++c) {
        pair<ll,ll> x=get_hash(c,l1,r1);
        pair<ll,ll> y=get_hash(c,l2,r2);
        if (x!=y) {
            ans+=char('a'+c);
        }
    }
    cout<<ans.size()<<endl;
    cout<<ans<<endl;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cin>>n>>q;
    cin>>s;
    init();
    while (q--) {
        int l1,r1,l2,r2;
        cin>>l1>>r1>>l2>>r2;
        solution(l1,r1,l2,r2);
    }
    return 0;
}