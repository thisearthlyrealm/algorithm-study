#include <iostream>
#include <vector>
#include <string>
using namespace std;
typedef long long ll;
bool checkrow(int n,int m,const vector<string>& graph,int sx,int sy,int dir) {
    int L=sx,R=sy;
    while (L-1>=0 && graph[sx][L-1]!='#') L--;
    while (R+1<m && graph[sx][R+1]!='#') R++;
    vector<int> ok(m,1);
    for (int r=sx+dir;r>=0 && r<n;r+=dir) {
        for (int c=L;c<=R;c++) {
            if (graph[r][c]=='#') ok[c]=0;
        }
        int c=L;
        while (c<=R) {
            if (graph[r][c]=='#') {
                c++;
                continue;
            }
            int left=-1,right=-1;
            while (c<=R && graph[r][c]!='#') {
                if (ok[c]) {
                    if (c<=sy) left=c;
                    if (c>=sy && right==-1) right=c;
                }
                c++;
            }
            if (left!=-1 && right!=-1 && left<right) return true;
        }
    }
    return false;
}
bool checkcol(int n,int m,const vector<string>& graph,int sx,int sy,int dir) {
    int U=sx,D=sx;
    int ans=0;
    return ans;
}