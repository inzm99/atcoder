# %% A

from typing import Counter


n,a,x,y = map(int,input().split())

if n <= a:
    print(x*n)
else:
    print(a*x + (n-a)*y)

# %% B

n = int(input())
s = len(str(int(input())))
if (n-s) %2 == 0:
    print('Takahashi')
else:
    print('Aoki')

# %% C
from collections import Counter
n,k = map(int,input().split())
c= list(map(int,input().split()))

ct = Counter(c[0:k])
ans = len(ct)

for i in range(n-k):
    ct[c[i]] -= 1
    ct[c[i+k]] += 1
    print(ct)
    ans = max(ans, len(ct))

print(ans)

# %% D