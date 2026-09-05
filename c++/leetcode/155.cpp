#include <iostream>
#include <stack>
#include <climits>
using namespace std;
class MinStack {
    stack<pair<int,int>> st;
public:
    MinStack() {
        st.emplace(0,INT_MAX);
    }
    void push(int value) {
        st.emplace(value,min(getMin(),value));
    }
    void pop() {
        st.pop();
    }
    int top() {
        return st.top().first;
    }
    int getMin() {
        return st.top().second;
    }
};
int main() {
    MinStack s;
    s.push(-2);
    s.push(0);
    s.push(-3);
    cout<<"getMin="<<s.getMin()<<endl;
    s.pop();
    cout<<"top="<<s.top()<<endl;
    cout<<"getMin="<<s.getMin()<<endl;
    return 0;
}
