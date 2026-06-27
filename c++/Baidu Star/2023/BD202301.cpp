#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
using namespace std;
typedef long long ll;
const int INF=1e9;
vector<int> bfs(int start,const vector<vector<int>>& graph) {
    int n=graph.size()-1;
    vector<int> dist(n+1,INF);
    queue<int> q;
    dist[start]=0;
    q.push(start);
    while (!q.empty()) {
        int u=q.front();
        q.pop();
        for (int v: graph[u]) {
            if (dist[v]==INF) {
                dist[v]=dist[u]+1;
                q.push(v);
            }
        }
    }
    return dist;
}
ll solution(ll TE,ll FE,ll S,int T,int F,int N,const vector<vector<int>>& graph) {
    vector<int> disT=bfs(T,graph);
    vector<int> disF=bfs(F,graph);
    vector<int> disN=bfs(N,graph);
    ll ans=INT_MAX;
    for (int i=1;i<=N;i++) {
        if (disT[i]==INF || disF[i]==INF || disN[i]==INF) continue;
        ll cost=disT[i]*TE+disF[i]*FE+disN[i]*(TE+FE-S);
        ans=min(ans,cost);
    }
    if (ans==INT_MAX) return -1;
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll TE,FE,S;
    cin>>TE>>FE>>S;
    int T,F,N,M;
    cin>>T>>F>>N>>M;
    vector<vector<int>> graph(N+1);
    for (int i=0;i<M;i++) {
        int x,y;
        cin>>x>>y;
        graph[x].push_back(y);
        graph[y].push_back(x);
    }
    const ll ans=solution(TE,FE,S,T,F,N,graph);
    cout<<ans<<endl;
    return 0;
}