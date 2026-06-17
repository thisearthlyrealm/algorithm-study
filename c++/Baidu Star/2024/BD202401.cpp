#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;
int solution(int N,ll B,const vector<ll>& P,const vector<ll>& S) {
    vector<pair<ll,int>> full(N);
    for (int i=0;i<N;i++) {
        full[i]={P[i]+S[i],i};
    }
    sort(full.begin(),full.end());
    int ans=0;
    for (int disc=0;disc<N;disc++) {
        ll cost=P[disc]/2+S[disc];
        if (cost>B) continue;
        ll rem=B-cost;
        int cnt=1;
        for (int j=0;j<N;j++) {
            int idx=full[j].second;
            if (idx == disc) continue;
            if (full[j].first<=rem){
                rem-=full[j].first;
                cnt++;
            }else {
                break;
            }
        }
        ans=max(ans,cnt);
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int N;
    ll B;
    cin>>N>>B;
    vector<ll> P(N),S(N);
    for (int i=0;i<N;i++) {
        cin>>P[i]>>S[i];
    }
    cout<<solution(N,B,P,S)<<"\n";
    return 0;
}