#include <iostream>
using namespace std;
int solution(int n) {
    if (n==0) return 1;
    int ans=0;
    for (int i=0;i<n;i++) {
        ans+=solution(i)*solution(n-1-i);
    }
    return ans;
}
int main() {
    int n;
    cin>>n;
    int ans=solution(n);
    cout<<ans<<endl;
    return 0;
}