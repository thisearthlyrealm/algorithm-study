#include <iostream>
#include <vector>
#include <string>
using namespace std;
int solution(string& s) {
    vector<int> m(128,0);
    int ans=0;
    int i=-1;
    for (int j=0;j<s.size();++j) {
        i=max(i,m[s[j]]);
        m[s[j]]=j;
        ans=max(ans,j-i);
    }
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin>>s;
    const int ans=solution(s);
    cout<<ans<<endl;
    return 0;
}