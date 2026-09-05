#include <iostream>
#include <vector>
using namespace std;
int solution(vector<int>& gas,vector<int>& cost) {
    int sum_gas=0;
    int sum_cost=0;
    for (int i=0;i<gas.size();i++) {
        sum_gas+=gas[i];
        sum_cost+=cost[i];
    }
    if (sum_gas<sum_cost) {
        return -1;
    }
    int ans=0;
    int start=0;
    for (int i=0;i<gas.size();i++) {
        ans+=gas[i]-cost[i];
        if (ans<0) {
            ans=0;
            start=i+1;
        }
    }
    return start;
}
int main() {
    vector<int> gas={1,2,3,4,5};
    vector<int> cost={3,4,5,1,2};
    int ans=solution(gas,cost);
    cout<<ans<<endl;
    return 0;
}