#include <iostream>
#include <vector>
using namespace std;
int solution(vector<int>& prices) {
    int n = prices.size();
    int ans=0;
    for (int i=0;i<n;i++) {
        if (prices[i]>=prices[i-1]) {
            ans+=prices[i]-prices[i-1];
        }
    }
    return ans;
}
int main() {
    vector<int> prices={7,1,5,3,6,4};
    int ans=solution(prices);
    cout<<ans<<endl;
    return 0;
}
