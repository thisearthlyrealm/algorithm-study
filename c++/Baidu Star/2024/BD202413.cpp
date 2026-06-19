#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;
void pre_solution(string& s,int l,int r) {
    for (int i=l;i<=r;++i) {
        s[i]=s[i]=='0'?'1':'0';
    }
}
int solution(string& s,int l,int r) {
    int ans1=0,ans2=0;
    for (int i=l;i<=r;++i) {
        char expect1=((i-l)%2==0)?'0':'1';
        char expect2=((i-l)%2==0)?'1':'0';
        if (s[i]!=expect1) ++ans1;
        if (s[i]!=expect2) ++ans2;
    }
    return min(ans1,ans2);
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n,q;
    cin>>n>>q;
    string s;
    cin>>s;
    while (q--) {
        int t,l,r;
        cin>>t>>l>>r;
        l--,r--;
        if (t==1) {
            pre_solution(s,l,r);
        }else {
            const int ans=solution(s,l,r);
            cout<<ans<<endl;
        }
    }
}