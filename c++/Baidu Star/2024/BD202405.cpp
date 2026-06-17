#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;
ll solution(int n,vector<ll>& nums) {
    if (n==1) return 0;
    vector<ll> nums1;
    vector<ll> nums2;
    ll max_int=nums[0],min_int=nums[0];
    for(int i=0;i<n;++i) {
        max_int=max(max_int,nums[i]);
        min_int=min(min_int,nums[i]);
        if (nums[i]>=0) {
            nums1.push_back(nums[i]);
        }else {
            nums2.push_back(nums[i]);
        }
    }
    sort(nums.begin(),nums.end());
    min_int=nums[0];
    max_int=nums[n-1];
    if (n==2) {
        ll ans=max_int-min_int;
        ll x=nums[0]>=0?nums[0]:-nums[0];
        ll y=nums[1]>=0?nums[1]:-nums[1];
        ans=max(ans,x);
        ans=max(ans,y);
        return ans;
    }
    vector<ll> pre(n+1,0);
    for(int i=1;i<=n;++i) {
        pre[i]=pre[i-1]+nums[i-1];
    }
    ll ans=max_int-min_int;
    ans=max(ans,max_int-2*min_int);
    ans=max(ans,2*max_int-min_int);
    for(int r=2;r<n;++r) {
        ll sum=pre[n]-pre[n-r];
        ans=max(ans,sum+max_int-min_int);
    }
    for(int l=2;l<n;++l) {
        ll sum=pre[l];
        ans=max(ans,max_int-sum-min_int);
    }
    const ll INF=(1LL<<62);
    vector<ll> best(n+1,INF);
    for(int i=2;i<=n;++i) {
        best[i]=min(best[i-1],pre[i]);
    }
    for(int r=2;r<=n-2;++r) {
        int limit=n-r;
        if (limit<2) continue;
        ll sum=pre[n]-pre[n-r];
        ans=max(ans,sum+max_int-best[limit]-min_int);
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    cin>>T;
    while(T--) {
        int n;
        cin>>n;
        vector<ll> nums(n);
        for(int i=0;i<n;++i) {
            cin>>nums[i];
        }
        const ll ans=solution(n,nums);
        cout<<ans<<endl;
    }
    return 0;
}