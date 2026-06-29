#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;
bool cmp(const pair<int,int>& a,const pair<int,int>& b) {
    if (a.first!=b.first) return a.first>b.first;
    return a.second<b.second;
}
int solution(int n,vector<pair<int,int>>& cats) {
    sort(cats.begin(),cats.end(),cmp);
    vector<int> speed;
    vector<int> cnts;
    int ans=0;
    for (int i=0;i<n;) {
        int pos=cats[i].first;
        int v=cats[i].second;
        int cnt=0;
        while (i<n && cats[i].first==pos) {
            v=min(v,cats[i].second);
            cnt++;
            i++;
        }
        if (speed.empty() || v<=speed.back()) {
            speed.push_back(v);
            cnts.push_back(cnt);
            ans=max(ans,cnt);
        }else {
            cnts.back()+=cnt;
            ans=max(ans,cnts.back());
        }
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    vector<pair<int,int>> cats(n);
    for (int i=0;i<n;i++) {
        cin>>cats[i].first>>cats[i].second;
    }
    const int ans=solution(n,cats);
    cout<<ans<<"\n";
}