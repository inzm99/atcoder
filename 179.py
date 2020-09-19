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
