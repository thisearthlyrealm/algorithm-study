#include <iostream>
#include <vector>
using namespace std;
typedef long long ll;
const int MOD=998244353;
void initprime(vector<bool>& not_prime,int n) {
    static bool initialized=false;
    if (initialized) return;
    initialized=true;
    not_prime[0]=true;
    not_prime[1]=true;
    for (int i=2;i*i<=n;++i) {
        if (not_prime[i]) continue;
        for (int j=i*i;j<=n;j+=i) {
            not_prime[j]=true;
        }
    }
}
vector<ll> get_inv(int n,ll p) {
    vector<ll> inv(n+1);
    inv[1]=1;
    for (int i=2;i<=n;++i) {
        inv[i]=(p-p/i)*inv[p%i]%p;
    }
    return inv;
}
int solution(int n) {
    vector<bool> not_prime(n+1,false);
    initprime(not_prime,n);
    ll lcm=1;
    for (int i=2;i<=n;++i) {
        if (!not_prime[i]) {
            ll pk=i;
            while (pk*i<=n) {
                pk*=i;
            }
            lcm=(lcm*pk)%MOD;
        }
    }
    vector<ll> inv=get_inv(n,MOD);
    ll sum=0;
    for (int i=1;i<=n;++i) {
        ll coeff=(n-2*i+1)%MOD;
        if (coeff<0) {
            coeff+=MOD;
        }
        sum=(sum+coeff*inv[i])%MOD;
    }
    ll ans=lcm*sum%MOD;
    return ans;
}
int main() {
    int n;
    cin>>n;
    cout<<solution(n);
    return 0;
}