#include <iostream>
#include <vector>
using namespace std;
typedef long long ll;
int solution(int N,int K,const vector<int>& P) {
    vector<int> last(1000001,-1);
    vector<int> vis(1000001,0);
    int ans=0;
    for (int i=1;i<=N;i++) {
        int x=P[i];
        if (last[x]!=-1 && i-last[x]<=K && !vis[x]) {
            ans^=x;
            vis[x]=1;
        }
        last[x]=i;
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int N,K;
    cin>>N>>K;
    vector<int> P(N+1);
    for (int i=1;i<=N;i++) {
        cin>>P[i];
    }
    const int ans=solution(N,K,P);
    cout<<ans<<endl;
    return 0;
}