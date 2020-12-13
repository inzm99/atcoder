#%% A

n,a,b=map(int,input().split())

print(n-a+b)

#%% B

n=int(input())
x = list(map(int,input().split()))

import numpy as np
import math

X = np.array(x)

X = np.abs(X)

print(sum(X))
print(math.sqrt(sum(X*X)))
print(max(X))

#%% C
import math
n=int(input())
ans=[]
for i in range(1,int(math.sqrt(n)) + 1):
    if n%i==0:
        ans.append(i)
        ans.append(n//i)
ans.sort()
for a in ans:
    print(a)