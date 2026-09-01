#include <iostream>
#include <string>
using namespace std;
int solution(string& s) {
    int n=s.length()-1;
    int i=n;
    while (s[i]==' ') {
        i--;
    }
    int j=i-1;
    while (s[j]!=' ') {
        j--;
    }
    return i-j;
}
int main() {
    string s="   fly me   to   the moon  ";
    int ans=solution(s);
    cout<<ans<<endl;
    return 0;
}