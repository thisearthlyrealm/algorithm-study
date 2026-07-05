#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
using namespace std;
typedef long long ll;
struct node {
    ll x,y,z;
};
ll getdist(const node& a, const node& b) {
    ll dx=a.x-b.x;
    ll dy=a.y-b.y;
    ll dz=a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
}
ll solution(int n,int k,const vector<ll>& c,const vector<node>& p) {
    vector<ll> dist(n+1,INT_MAX);
    vector<int> visited(n+1,0);
    vector<ll> edge;
    dist[1]=0;
    ll sum=0;
    for (int i=1;i<=n;i++) {
        int u=-1;
        for (int j=1;j<=n;j++) {
            if (!visited[j] && (u==-1 || dist[j]<dist[u])) {
                u=j;
            }
        }
        visited[u]=1;
        sum+=dist[u];
        if (dist[u]!=0) {
            edge.push_back(dist[u]);
        }
        for (int v=1;v<=n;v++) {
            if (!visited[v]) {
                ll w=getdist(p[u],p[v]);
                if (w<dist[v]) {
                    dist[v]=w;
                }
            }
        }
    }
    sort(edge.rbegin(),edge.rend());
    vector<ll> pre(n,0);
    for (int i=1;i<n;i++) {
        pre[i]=pre[i-1]+edge[i-1];
    }
    ll ans=INT_MAX;
    for (int i=0;i<=n-k;i++) {
        int cnt=k+i;
        ll cost=sum-pre[cnt-1]+c[i];
        ans=min(ans,cost);
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n,k;
    cin>>n>>k;
    vector<ll> c(n-k+1,0);
    for (int i=1;i<=n-k;i++) {
        cin>>c[i];
    }
    vector<node> p(n+1);
    for (int i=1;i<=n;i++) {
        cin>>p[i].x>>p[i].y>>p[i].z;
    }
    const ll ans=solution(n,k,c,p);
    cout<<ans<<"\n";
    return 0;
}