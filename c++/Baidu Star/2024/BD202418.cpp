#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
typedef long long ll;
void pre_solution(vector<pair<ll,int>>& diff,ll& begin,ll x,char c) {
    ll l,r;
    if (c=='R') {
        l=begin;
        r=begin+x-1;
        begin=r;
    }else {
        l=begin-x+1;
        r=begin;
        begin=l;
    }
    diff.emplace_back(l,1);
    diff.emplace_back(r+1,-1);
}
ll solution(vector<pair<ll,int>>& diff) {
    sort(diff.begin(),diff.end());
    ll ans=0;
    ll cur=0;
    ll last=diff[0].first;
    for (int i=0;i<(int)diff.size();) {
        ll now=diff[i].first;
        if (cur%4==1) {
            ans+=now-last;
        }
        while (i<(int)diff.size() && diff[i].first==now) {
            cur+=diff[i].second;
            ++i;
        }
        last=now;
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<pair<ll,int>> diff;
    int N;
    cin>>N;
    ll begin=0;
    while (N--) {
        ll x;
        char c;
        cin>>x>>c;
        pre_solution(diff,begin,x,c);
    }
    const ll ans=solution(diff);
    cout<<ans<<endl;
    return 0;
}