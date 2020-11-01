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
        

# AC

import sys

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 7)


class SegTree:
    """
    Segment Tree
    """

    def __init__(self, init_val, segfunc, ide_ele):
        """
        初期化

        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        """
        k番目の値をxに更新 O(N)

        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る O(logN)

        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res


def main():
    N, K = map(int, input().split())
    A = []
    MAX = 300005
    for i in range(N):
        A.append(int(input()))

    ide_ele = 0

    dp = SegTree([0] * MAX, segfunc=max, ide_ele=ide_ele)

    for i in range(N):
        a = A[i]
        l = max(0, a - K)
        r = min(MAX, a + K + 1)
        now = dp.query(l, r) + 1
        dp.update(a, now)

    print(dp.query(0, MAX))


if __name__ == "__main__":
    main()

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
