#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
using namespace std;
int solution(string& s) {
    unordered_map<char,int> a={{'(',1},{')',-1}};
    int n=s.size();
    vector<int> pre_sum(n+1,0);
    for (int i=1;i<=n;++i) {
        pre_sum[i]=pre_sum[i-1]+a[s[i-1]];
    }
    vector<int> pre_min(n+1,0);
    pre_min[0]=pre_sum[0];
    for (int i=1;i<=n;++i) {
        pre_min[i]=min(pre_min[i-1],pre_sum[i]);
    }
    vector<int> suf_min(n+1,0);
    suf_min[n]=pre_sum[n];
    for (int i=n-1;i>=0;--i) {
        suf_min[i]=min(suf_min[i+1],pre_sum[i]);
    }
    if (pre_sum[n]==0) return 0;
    int ans=0;
    int delta;
    char need;
    if (pre_sum[n]==2) {
        delta=-2;
        need='(';
    }else {
        delta=2;
        need=')';
    }
    for (int i=1;i<=n;++i) {
        if (s[i-1]!=need) continue;
        if (pre_min[i-1]<0) continue;
        if (suf_min[i]+delta<0) continue;
        ++ans;
    }
    return ans;
}
int main() {
    string s;
    cin>>s;
    const int ans=solution(s);
    cout<<ans<<endl;
    return 0;
}