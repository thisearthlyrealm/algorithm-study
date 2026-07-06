#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;
typedef long long ll;
int lower_bound_search(const vector<int>& nums,int target) {
    int l=0,r=nums.size();
    while(l<r) {
        int mid=l+(r-l)/2;
        if (nums[mid]>=target) {
            r=mid;
        }else {
            l=mid+1;
        }
    }
    if (l==nums.size()) return -1;
    return l;
}
int solution(const vector<int>& apple,vector<int>& capacity) {
    sort(capacity.begin(), capacity.end(),greater<int>());
    int sum=accumulate(apple.begin(),apple.end(),0);
    vector<int> pre_sum(capacity.size()+1,0);
    for (int i=1;i<=capacity.size();i++) {
        pre_sum[i]=pre_sum[i-1]+capacity[i-1];
    }
    int ans=lower_bound_search(pre_sum,sum);
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<int> apple={1,3,2};
    vector<int> capacity={4,3,1,5,2};
    const int ans=solution(apple,capacity);
    cout<<ans<<endl;
    return 0;
}