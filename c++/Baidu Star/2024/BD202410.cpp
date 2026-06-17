#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
void pre_solution(vector<int>& diff,int l,int k) {
    diff[l]+=1;
    diff[k+1]-=1;
}
double solution(vector<int>& diff,int N) {
    vector<int> pre_ans(N+1,0);
    for (int i=1;i<=N;i++) {
        pre_ans[i]=pre_ans[i-1]+diff[i];
    }
    pre_ans.erase(pre_ans.begin());
    sort(pre_ans.begin(),pre_ans.end());
    if (N%2==1) {
        return pre_ans[N/2];
    }else {
        return (pre_ans[N/2-1]+pre_ans[N/2])/2.0;
    }
}
int main() {
    int N,K;
    cin>>N>>K;
    vector<int> diff(N+2,0);
    while (K--) {
        int A,B;
        cin>>A>>B;
        pre_solution(diff,A,B);
    }
    const double ans=solution(diff,N);
    cout<<ans<<endl;
    return 0;
}