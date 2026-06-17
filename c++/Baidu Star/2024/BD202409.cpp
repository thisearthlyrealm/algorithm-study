#include <iostream>
#include <vector>
#include <string>
using namespace std;
string solution(int n,string& s) {
    vector<int> pre(n+1,0);
    for (int i=1;i<=n;++i) {
        pre[i]=(pre[i-1]*2+(s[i-1]-'0'))%3;
    }
    string ans;
    for (int i=1;i<=n;++i) {
        int pos=n-i;
        int cnt=(pre[pos]+(s[pos]-'0'))%3;
        if ((n-i)%2==0) {
            string t="ACB";
            ans+=t[cnt];
        }else {
            string t="ABC";
            ans+=t[cnt];
        }
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    string s;
    cin>>s;
    const string ans=solution(n,s);
    cout<<ans<<endl;
    return 0;
}