# %% A

# a,b,c = map(int,input().split())

# if c == 0:
#     if a > b:
#         print('Takahashi')
#     else:
#         print('Aoki')

# if c == 1:
#     if b > a:
#         print('Aoki')
#     else:
#         print('Takahashi')

# %% B

# n,s,d = map(int,input().split())
# ans = 'No'
# for i in range(n):
#     x, y = map(int,input().split())
#     if x < s and y > d:
#         ans = 'Yes'

# print(ans)

# %% C
from itertools import product

n,m =  map(int,input().split())
ab = []
cd = []
ans = 0
for i in range(m):
    ab.append(list(map(int,input().split())))
k = int(input())
for i in range(k):
    cd.append(list(map(int,input().split())))

for balls in product(*cd): # unpack [[1, 2], [1, 3], [2, 3]] -> [1, 2] [1, 3] [2, 3]
    balls = set(balls)
    cnt = sum(a in balls and b in balls for a, b in ab)
    ans = max(ans, cnt)

print(ans)

# %% D
import math

N = int(input())
ans = 0
# N:Sum = ( Start + end:(Start + Length - 1)) Length / 2
# ( 2S + L - 1 ) * L = 2N
# S = ( 2N/L - L + 1) / 2
# ( 2N/L - L + 1): 偶数
# L: 2Nの約数

for L in range(1, int(math.sqrt(2*N)) + 1):
    if (2*N) % L != 0:
        continue
    Y = (2*N) // L - L + 1
    if Y % 2 == 0:
        # print(L, Y/2)
        ans += 1

print(ans*2)



