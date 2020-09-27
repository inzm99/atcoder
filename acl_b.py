# %%

K= int(input())
print('ACL'*K)

# %%

a,b,c,d = map(int,input().split())

if b>=c and a<=d:
    print('Yes')
elif b<=c and a>=d:
    print('Yes')
else:
    print('No')

# %%

class UnionFind:
    def __init__(self, n):
        self.par = list(range(n)) #親
        self.rank = [0] * n #根の深さ

    # xの属する根を求める
    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            # if they have the same parent, no need to union
            return
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    # xとyが同じ集合に属するかを判定（同じ根に属するか）
    def same(self, x, y):
        return self.find(x) == self.find(y)


N, M = map(int, input().split())
uf = UnionFind(N)

for _ in range(M):
    a, b = map(int, input().split())
    uf.unite(a-1, b-1)

ans = -1
for i in range(N):
    if uf.par[i] == i:
        ans +=1
print(ans)

# %%
# D
# dp # segtree

n,k = map(int,input().split())
a=[]
for i in range(n):
    a.append(int(input()))
    
dp= [0]*(n+1)
# dp[0]=1
maxa = 0

for i in range(1,n):
    tmp = 0
    for j in range(i,maxa-1,-1):
        if abs(a[i] - a[j]) <= k:
            tmp = max(tmp,dp[j])
        if maxa == tmp:
            dp[i]=tmp+1
            maxa+=1
            # print('break')
            break
    dp[i] = tmp + 1
print(dp[n-1])
        
# %%
# E
# 

import numpy as np

mod = 998244353
n,q = map(int,input().split())
l=[0]*q
r=[0]*q
d=[0]*q
for i in range(q):
    l[i],r[i],d[i]=map(int,input().split())


s = np.array([1]*n)
for i in range(q):
    s[l[i]-1:r[i]] = d[i]
    ans = ''
    for j in range(n):
        ans += str(s[j])
    print(int(ans)%mod)
# %%
