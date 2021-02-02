# %% A
x,y=map(int,input().split())
if abs(x-y) <=2:
    print('Yes')
else:
    print('No')

# %% B

import numpy as np
n =int(input())
a = np.array(list(map(int,input().split())))
b = np.array(list(map(int,input().split())))

if np.dot(a,b) ==0:
    print('Yes')
else:
    print('No')

# %% C

n = int(input())
a = list(map(int,input().split()))

half = 2**(n-1)
max_left = 0
max_left_i = 0
max_right = 0
max_right_i = 0

for i in range(0,half):
    if max_left < a[i]:
        max_left = a[i]
        max_left_i = i+1


for i in range(half,len(a)):
    if max_right < a[i]:
        max_right = a[i]
        max_right_i = i+1

if max_left > max_right:
    print(max_right_i)
else:
    print(max_left_i)

# %% D

n, prime = map(int,input().split())

event =[]

for i in range(n):
    a,b,c = map(int,input().split())
    b += 1
    event.append((a, c))
    event.append((b,-c))

event.sort()

ans = 0
t = 0
fee = 0

for x, y in event:
    if x != t:
        ans += min(prime, fee) * (x - t)
        t = x
    fee += y

print(ans)