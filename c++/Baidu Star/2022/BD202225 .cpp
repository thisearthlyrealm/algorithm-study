#include <iostream>
#include <vector>
using namespace std;
typedef long long ll;
ll solution(int n,const vector<ll>& nums) {
    ll ans=0;
    for (int i=1;i<=n;++i) {
        ans+=nums[i];
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    vector<ll> nums(n+1);
    for (int i=1;i<=n;++i) {
        cin>>nums[i];
    }
    const ll ans=solution(n,nums);
    cout<<ans<<endl;
    return 0;
}