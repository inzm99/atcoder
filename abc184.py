# %%

a,b = map(int,input().split())
c,d = map(int,input().split())

print(a*d - b*c)

# %%

n,x= map(int,input().split())
S = list(input())

for s in S:
    if s=='o':
        x += 1
    elif x==0:
        pass
    else:
        x -= 1

print(x)


# %%

import sys

r1, c1 = map(int,input().split())
r2, c2 = map(int,input().split())

def judge(a,b, c,d):
    if a+b == c+d:
        return True
    if a-b == c-d:
        return True
    if abs(a-c) + abs(b-d) <= 3:
        return True
    return False

if r1==r2 and c1 ==c2:
    print(0)
    sys.exit()
elif judge(r1,c1,r2,c2):
    print(1)
    sys.exit()
else:
    if (r1+c1) %2 == (r2+c2)%2:
        print(2)
        sys.exit()
    for x in range(-2,3):
        for y in range(-2,3):
            if judge(r1+x,c1+y,r2,c2):
                print(2)
                sys.exit()
        for x,y in ((-3,0),(3,0),(0,3),(0,-3)):
            if judge(r1+x,c1+x,r2,c2):
                print(2)
                sys.exit()

    print(3)

# %%

# 動的計画法（DP）
# dp[a][b][c] = 金a銀b銅c枚ある状態からの期待値

A,B,C = map(int, input().split())

DP = [[[0]*101 for _ in range(101)] for _ in range(101)]

for a in range(100)[::-1]:
    for b in range(100)[::-1]:
        for c in range(100)[::-1]:
            s = a + b + c
            if s>0:
                DP[a][b][c] = 1 + (a * DP[a+1][b][c] + b * DP[a][b+1][c] + c * DP[a][b][c+1]) / s

print(DP[A][B][C])

# %%

# BFS

