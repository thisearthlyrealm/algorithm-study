#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;
typedef long long ll;
const int MOD=998244353;
ll qpow(ll a,ll b) {
    ll res=1;
    a%=MOD;
    while (b) {
        if (b&1) res=res*a%MOD;
        a=a*a%MOD;
        b>>=1;
    }
    return res;
}
void add_edge(int u,int v,int id,int idx,vector<int>& to,vector<int>& eid,vector<int>& nxt,vector<int>& head) {
    to[++idx]=v;
    eid[idx]=id;
    nxt[idx]=head[u];
    head[u]=idx;
}
ll solution(int n,int m,ll k,vector<int>& eu,vector<int>& ev) {
    vector<int> head,to,nxt,eid;
    vector<int> dfn,low,belong,comp_size;
    vector<char> is_bridge;
    int idx=0,time_cnt=0,comp_cnt=0;
}