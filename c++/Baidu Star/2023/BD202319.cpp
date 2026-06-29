#include <iostream>
#include <vector>
#include <string>
using namespace std;
typedef long long ll;
string solution(int n) {
    vector<int> cnt(n+1,0);
    for (int i=1;i<=n;i++) {
        cnt[i]=n+1-i;
    }
    vector<int> prime;
    vector<int> noi_prime(n+1,0);
    for (int i=2;i<=n;i++) {
        if (!noi_prime[i]) prime.push_back(i);
        for (int j=0;j<prime.size() && i*prime[j]<=n;j++) {
            noi_prime[i*prime[j]]=1;
            if (i%prime[j]==0) break;
        }
    }
    string ans="f("+to_string(n)+")=";
    bool first=true;
    for (int i=0;i<prime.size();i++) {
        int p=prime[i];
        ll num=0;
        for (int j=p;j<=n;j+=p) {
            int t=j;
            while (t%p==0) {
                num+=cnt[j];
                t/=p;
            }
        }
        if (!first) ans+="*";
        first=false;
        ans+=to_string(p);
        if (num>1) {
            ans+="^";
            ans+=to_string(num);
        }
    }
    if (first) ans+="1";
    return ans;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin>>n;
    const string ans=solution(n);
    cout<<ans<<"\n";
    return 0;
}