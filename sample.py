# %%
## bit全探索

from itertools import product

N = 3

for p in product((0, 1), repeat=N):
    print(p)

# %%
# 二分探索

from bisect import bisect_right

def search(t, i):
    """
    t: list 探索元の数列
    i: int 探索する値
    """
    ix = bisect_right(t, i)
    if t[ix-1] != i:
        return False
    return True

t = [1,3,5,6,7,10]
i = 7

t.sort()

print(search(t, i))  # True

# %%
# BFS

# ABC007 C - 幅優先探索
from collections import deque

R, C = map(int, input().split()) # 行数　列数
sy, sx = map(int, input().split()) # スタート地点の座標
gy, gx = map(int, input().split()) # ゴール地点の座標
sy, sx, gy, gx = sy-1, sx-1, gy-1, gx-1
c = [[c for c in input()] for _ in range(R)] # 文字が . なら空きマス、# なら壁マス
visited = [[-1]*C for _ in range(R)]


def bfs(sy,sx,gy,gx,c,visited):
    visited[sy][sx] = 0
    Q = deque([]) # 探索リスト
    Q.append([sy, sx])
    while Q:
        y,x = Q.popleft()

        if [y, x] == [gy, gx]:
            return visited[y][x]

        for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]: # 探索リストの上下左右を探索
            if c[y+i][x+j] == '.' and visited[y+i][x+j] == -1: # 探索可能かつ未探索の場合
                visited[y+i][x+j] = visited[y][x]+1 # ＋１歩
                Q.append([y+i,x+j]) # 探索リストに追加


print(bfs(sy, sx, gy, gx, c, visited))

# %%
## DFS

# ABC138 D - Ki

import sys
sys.setrecursionlimit(10**7)

N,Q = map(int,input().split())
tree = [[] for _ in range(N)]

# tree[i]: i番目の頂点とつながっている頂点（リスト）を入れる
for _ in range(N-1):
    a,b = map(int,input().split())
    tree[a-1].append(b-1)
    tree[b-1].append(a-1)

# x_list[i]: iの配下に足す数を入れる
x_list = [0]*N
for _ in range(Q):
    p,x = map(int,input().split())
    x_list[p-1] += x

ans = [0]*N

def dfs(child, parent):
    ans[child] = ans[parent] + x_list[child]
    for v in tree[child]:
        if v != parent:
            dfs(v, child)

dfs(0,0)
print(' '.join([str(i) for i in ans]))

# %%
## しゃくとり法

N=int(input())
a=list(map(int,input().split()))
a.append(-1)

ans = 0
cnt = 0
pre = -1

for i in range(N+1):
    if a[i] > pre:
        cnt += 1
    else:
        ans += cnt*(cnt+1)//2
        cnt = 1
    pre = a[i]

# for right_idx in range(N):
#     if a[right_idx] >= left:
#         cnt += 1
#     else:
#         ans += (cnt+cnt-1)//2
#         cnt = 1
#     left = a[right_idx]

# for left in range(N):
#     cnt = 2
#     right = left+1
#     if a[left] >= a[left+1]:
#         ans+=1
#     else:
#         while right < N and a[right-1] < a[right]:
#             right += 1
#             cnt += 1
#         ans += (cnt + cnt -1) // 2
        
    # print(a[left:right])
print(ans)



# %%
# いもす法



# %%
# ダイクストラ法

N, M = map(int,input().split())
C = [[100]*N for _ in range(N)]

for _ in range(M):
    a,b = map(int,input().split())
    a,b = a-1, b-1
    C[a][b] = 1
    C[b][a] = 1

for i in range(N):
    C[i][i] = 0

# 各ノードからすべてのノードへの最短距離を計算
for k in range(N):
    for i in range(N):
        for j in range(N):
            C[i][j] = min(C[i][j], C[i][k]+C[j][k])

for c in C:
    print(c.count(2))

# %%
# 貪欲法

# N日間のうち、K日間働く。ある日働いたら、その直後のC日間は働かない。
N, K, C = map(int, input().split())
# S[i] が x の時は働かない
S = list(input())
rS = S[::-1]
lcnt, rcnt = 0, 0
left, right = [0]*N, [0]*N
llimit, rlimit = -1, -1

for i in range(N):
    # 左から貪欲
    # llimit (数字がカウントされた日+C日)以降
    # 上記かつ、シフトに入れる日 ('o')であればシフトに入る。
    if S[i] == 'o' and llimit<i:
        lcnt += 1
        left[i] = lcnt
        llimit = i + C

    # 右から貪欲
    if rS[i] == 'o' and rlimit<i:
        rcnt += 1
        right[-(i + 1)] = K + 1 - rcnt
        rlimit = i + C

#print(left)
#print(right)

# leftとrightで同じ数字が入っている日にちを出力
for i in range(N):
    if left[i]==right[i] and left[i]>0:
        print(i + 1)

# %%
# 動的計画法（DP）

N, M = map(int, input().split())
# Broken stairs
A = set([int(input()) for _ in range(M)])
mod = 10**9+7

# How many are there to climb
step = [0] * (N+1)
step[0] = 1
step[1] = 1

for i in range(N+1):
    if i==0 or i==1:
        if i in A:
            step[i] = 0
    else:
        step[i] = step[i-1]+step[i-2]
        if i in A:
            step[i] = 0
        step[i] %= mod

print(step[-1])


# %%
# 優先度付きキュー

import heapq
 
n, m = map(int,input().split())
a = [-int(i) for i in input().split()]
heapq.heapify(a) 
 
for i in range(m):
        heapq.heappush(a,heapq.heappop(a)/2)
 
print(sum([int(i) for i in a])*-1)


# %%
# Union Find

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

# print(len(set(uf.par)) - 1)

ans = -1
for i in range(N):
    if uf.par[i] == i:
        ans +=1
print(ans)


# %%


# %%
