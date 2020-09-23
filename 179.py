# %%

S = input()

print(S+'es' if S[-1]=='s' else S+'s')

# %%

n=int(input())

ans = 'No'
zoro = 0
for i in range(n):
    d1,d2 = map(int,input().split())
    if d1==d2:
        zoro+=1
    else:
        zoro=0
    if zoro ==3:
        ans='Yes'


print(ans)

# %%

import math
n= int(input())
ans=0
nn = int((n-1)//2)+1
for i in range(2,nn):
    for j in range(i,nn):
        if i*j>n-1:
            break
        else:
            if i==j:
                ans += 1
            else:
                ans+=2
# a or b =1
ans += (n-1-1)*2
print(ans+1)
# for k in range(1,nn):
#     if k*k >n-1:
#         break
#     else:
#         ans += 1

# nn = int(math.sqrt(n-1))+1
# for c in range(1,n):
#     for a in range(1,nn):
#         if (n-c)%a==0:
#             # ans+=1
#             if a == (n-c)//a:
#                 ans+=1
#             else:
#                 ans+=2
            # print(a,(n-c)//a,c)

# %%
# 配るDP（計算量NG）
n,k =map(int,input().split())
s =[]
for i in range(k):
    l,r=map(int,input().split())
    s+=(list(range(l,r+1)))
S= set(s)

mod = 998244353
step = [0] * (n+1)

step[0]=1
for i in range(n):
    for s in S:
        if i+s<=n:
            step[i+s]=(step[i]+step[i+s])%mod
        else:
            break

print(step[n-1]%mod)

# %%
# もらうDP
mod = 998244353
n,k =map(int,input().split())
step =[0]*(n+1)
# step[0]=1
step[1]=1
stepsum=[0]*(n+1)
stepsum[1]=1
l=[0]*k
r=[0]*k
for i in range(k):
    l[i],r[i]=map(int,input().split())

for i in range(2,n+1):
    for j in range(k):
        li = i - r[j]
        ri = i - l[j]
        if ri <= 0:
            continue
        # li = max(1,li)
        step[i] += stepsum[ri] - stepsum[max(0,li-1)]
        # step[i] %= mod
        # print(step)
    stepsum[i] = ( stepsum[i-1] + step[i] )%mod

print(step[n]%mod)

# %%

N,X,M=map(int,input().split())

ans=0
sumlist = [X,]
checklist = [-1]*(M+1)
A = X
# checklist[A] = 0

def calcans(start, end, sumlist):
    cnt = end - start + 1
    loop = (N - start - 1) // cnt
    add = N - (cnt * loop) - start
    return (sumlist[end]-sumlist[start-1])*loop + sumlist[start-1+add]

for i in range(1,N):
    A = (A**2) %M
    # print(A)
    if checklist[A-1] != -1:
        # print(f'start: {checklist[A]} end: {i-1}')
        ans = calcans(checklist[A-1], i-1, sumlist)
        # looplen = i-1 - checklist[A] + 1
        # looptimes = (N - checklist[A]-1) // looplen
        # ans += (sumlist[i-1] - sumlist[checklist[A]-1] )* looptimes
        # ans += sumlist[(N-checklist[A]) % looplen + checklist[A]-1]
        # print(f'loop end: {looplen}, {looptimes}')
        break
    elif A == 0:
        ans = sumlist[i-1]
        break
    else:
        checklist[A-1] = i
        sumlist.append(sumlist[-1] + A)
else:
    ans = sumlist[-1]

print(ans)

# %%
