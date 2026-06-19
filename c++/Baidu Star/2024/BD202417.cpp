#include <iostream>
#include <vector>
using namespace std;
typedef long long ll;
int n;
ll l,r,c;
vector<ll> a,b;
__int128 get_count(ll t) {
    if (t<=0) return 0;
    __int128 sum=0;
    for (int i=0;i<n;++i) {
        if (t<b[i]) continue;
        sum+=(t-b[i])/a[i]+1;
    }
    return sum;
}
ll solution() {
    __int128 base=get_count(l-1);
    if (get_count(r)-base<c) {
        return -1;
    }
    ll left=l,right=r;
    ll ans=r;
    while (left<=right) {
        ll mid=left+(right-left)/2;
        if (get_count(mid)-base>=c) {
            ans=mid;
            right=mid-1;
        }else {
            left=mid+1;
        }
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cin>>n;
    a.resize(n);
    b.resize(n);
    for (int i=0;i<n;++i) cin>>a[i];
    for (int i=0;i<n;++i) cin>>b[i];
    cin>>l>>r>>c;
    ll ans=solution();
    cout<<ans<<endl;
    return 0;
}