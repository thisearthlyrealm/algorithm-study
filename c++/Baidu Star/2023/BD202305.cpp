#include <iostream>
using namespace std;
typedef long long ll;
ll solution(ll p,ll k) {
    if (k==0) return 0;
    if (p==1) return 1;
    return k-(k-1)/p;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    cin>>T;
    while(T--) {
        ll p,k;
        cin>>p>>k;
        cout<<solution(p,k)<<"\n";
    }
    return 0;
}