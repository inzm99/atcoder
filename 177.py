# %%

d,t,s = map(int,input().split())

print('Yes' if d/s <= t else 'No')
# %%

s = input()
t = input()

maxt = len(t)

for i in range(0,len(s) - len(t) +1):
    s_part = s[i:i+len(t)+1]
    n=0
    for tt,ss in zip(t,s_part):
        if tt==ss:
            n+=1
    maxt =min(maxt,len(t)-n)
print(maxt)

# %%

n=int(input())
a=list(map(int,input().split()))
mod = 10**9+7

ans = 0
aa = 0
for i in range(n-1):
    aa += a.pop()
    # print(a,aa)
    ans +=  (aa * a[-1]) % mod
    ans %= mod

print(ans%mod)

# %%
import sys

sys.setrecursionlimit(1000000)

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

    def size(self, x):
        return -self.rank[self.find(x)]

N, M = map(int, input().split())
uf = UnionFind(N)

for _ in range(M):
    a, b = map(int, input().split())
    uf.unite(a-1, b-1)

# ans = 0

# for i in range(n):
#     ans = max(ans, uf.size(i))

# print(ans)

from collections import Counter
h = [uf.find(i) for i in range(N)]
c = Counter(h).most_common
print(c[0][1])

print(max(c.values()))

cnt = [0] *n
for i in range(N):
    cnt[uf.par[i]] += 1
print(max(cnt))