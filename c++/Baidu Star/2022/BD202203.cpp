#include <iostream>
#include <vector>
#include <climits>
using namespace std;
typedef long long ll;
void add_state(vector<pair<ll,ll>>& state,pair<ll,ll> now) {
    for (int i=0;i<state.size();i++) {
        if (state[i].first<=now.first && state[i].second>=now.second) {
            return;
        }
    }
    vector<pair<ll,ll>> temp;
    for (int i=0;i<state.size();++i) {
        if (now.first<=state[i].first && now.second>=state[i].second) {
            continue;
        }
        temp.push_back(state[i]);
    }
    temp.push_back(now);
    state=temp;
}
bool solution(int n,const vector<int>& a) {
    if (n<2) return false;
    vector<vector<pair<ll,ll>>> dp(n+1);
    dp[0].emplace_back(INT_MIN,INT_MAX);
    for (int i=1;i<=n;++i) {
        ll x=a[i];
        for (int j=0;j<dp[i-1].size();++j) {
            ll up=dp[i-1][j].first;
            ll down=dp[i-1][j].second;
            if (x>=up) {
                add_state(dp[i],{x,down});
            }
            if (x<=down) {
                add_state(dp[i],{up,x});
            }
        }
        if (dp[i].empty()) {
            return false;
        }
    }
    return !dp[n].empty();
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
    if (solution(n,a)) {
        cout<<"yes"<<"\n";
    }else {
        cout<<"no"<<"\n";
    }
    return 0;
}