#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;
typedef long long ll;
int bfs(int n,const vector<int> next) {
    vector<int> dist(n+1,INT_MAX);
    queue<int> q;
    dist[1]=0;
    q.push(1);
    while(!q.empty()) {
        int u=q.front();
        q.pop();
        if (u>1 && dist[u-1]==INT_MAX) {
            dist[u-1]=dist[u]+1;
            q.push(u-1);
        }
        if (u<n && dist[u+1]==INT_MAX) {
            dist[u+1]=dist[u]+1;
            q.push(u+1);
        }
        if (next[u]!=0 && dist[next[u]]==INT_MAX) {
            dist[next[u]]=dist[u]+1;
            q.push(next[u]);
        }
    }
    return dist[n];
}
int solution(int n,const vector<int>& a) {
    vector<int> next(n+1,0);
    vector<int> last(1000001,0);
    for (int i=n;i>=1;i--) {
        next[i]=last[a[i]];
        last[a[i]]=i;
    }
    return bfs(n,next);
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    vector<int> a(n+1);
    for (int i=1;i<=n;i++) {
        cin>>a[i];
    }
    const ll ans=solution(n,a);
    cout<<ans<<"\n";
    return 0;
}