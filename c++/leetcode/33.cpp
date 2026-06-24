#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
int findMinIndex(vector<int>& nums) {
    int left=-1,right=(int)(nums.size()-1);
    while (left+1<right) {
        int mid=left+(right-left)/2;
        if (nums[mid]<nums.back()) {
            right=mid;
        }else {
            left=mid;
        }
    }
    return right;
}
int binarySearch(vector<int>& nums,int left,int right,int target) {
    while (left+1<right) {
        int mid=left+(right-left)/2;
        if (nums[mid]>=target) {
            right=mid;
        }else {
            left=mid;
        }
    }
    if (right<(int)nums.size() && nums[right]==target)
        return right;
    return -1;
}
int solution(vector<int>& nums,int target) {
    int minIdx=findMinIndex(nums);
    if (target>nums.back()) {
        return binarySearch(nums,-1,minIdx,target);
    }else {
        return binarySearch(nums,minIdx-1,(int)nums.size(),target);
    }
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector<int> nums={4,5,6,7,0,1,2};
    int target;
    cin>>target;
    const int ans=solution(nums,target);
    cout<<ans<<endl;
    return 0;
}