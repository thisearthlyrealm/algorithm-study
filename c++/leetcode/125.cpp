#include <iostream>
#include <string>
using namespace std;
bool solution(string& s) {
    int n=s.size();
    int i=0;
    int j=n-1;
    while (i<j) {
        if (!isalpha(s[i])) {
            i++;
        }else if (!isalpha(s[j])) {
            j--;
        }else if (tolower(s[i])==tolower(s[j])) {
            i++;
            j--;
        }else {
            return false;
        }
    }
    return true;
}
int main() {
    string s="A man, a plan, a canal: Panama";
    bool ans=solution(s);
    cout<<boolalpha<<ans<<endl;
    return 0;
}
