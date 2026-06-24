#include <iostream>
using namespace std;
bool solution(int x) {
    if (x<0 || (x%10==0 && x!=0)) {
        return false;
    }
    int reverse_x=0;
    while (x>reverse_x) {
        reverse_x=reverse_x*10+x%10;
        x=x/10;
    }
    return x==reverse_x || x==reverse_x/10;
}
int main() {
    int x;
    cin>>x;
    const bool ans=solution(x);
    cout<<boolalpha<<ans<<endl;
    return 0;
}