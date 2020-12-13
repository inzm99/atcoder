# %%

# print(min(1,2,3,4))

# %%

n,m,t = map(int,input().split())
nmax = n
a = [0] * (m+1)
b = [0] * (m+1)

for i in range(1,m+1):
    a[i], b[i] = map(int,input().split())
    # b[i] = tmp - a[i]

# n = n-a[0]
ans = 'Yes'

for i in range(1,m+1):
    n = n - (a[i] - b[i-1])
    if n <= 0:
        ans= 'No'
        break
    n = n + (b[i] - a[i])
    n = min(nmax, n)
else:
    n = n - (t-b[m])
    if n <= 0:
        ans = 'No'

print(ans)

# %%
import math
def comb(n,r):
    return math.factorial(n) // (math.factorial(n-r) * math.factorial(r))

n= int(input())
print(comb(n-1, 11))

# %%
import math
import sys
n,m = map(int,input().split())
a = list(map(int,input().split()))
a = [0] + a + [n+1]

if m == 0:
    print(1)
    sys.exit()
a.sort()
b = []
# print(a)
mina=n
for i in range(m+1):
    b.append(a[i+1] - a[i] -1)
    if b[-1] != 0:
        mina = min(mina,  b[-1])
ans = 0
for bb in b:
    ans += math.ceil(bb/mina)

# print(b, mina)

print(ans)


# %%
