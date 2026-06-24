#include <iostream>
#include <unordered_map>
#include <stack>
#include <string>
using namespace std;
bool solution(string& s) {
    unordered_map<char,char> mapping={
        {')','('},
        {']','['},
        {'}','{'}
    };
    stack<char> st;
    for (int i=0;i<s.length();++i) {
        if (mapping.count(s[i])) {
            if (st.empty()) {
                return false;
            }
            char top=st.top();
            if (top!=mapping[s[i]]) {
                return false;
            }
            st.pop();
        }else {
            st.push(s[i]);
        }
    }
    return st.empty();
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin>>s;
    const bool ans=solution(s);
    cout<<boolalpha<<ans<<endl;
    return 0;
}