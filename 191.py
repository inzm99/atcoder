# %% A

# v,t,s,d = map(int,input().split())

# if t <= d/v <= s:
#     print('No')
# else:
#     print('Yes')


# # %% B

# n,x = map(int,input().split())
# a = list(map(int,input().split()))

# ans = []
# for i in a:
#     if i != x:
#         ans.append(i)

# print(*ans)

# %% C

h,w = map(int,input().split())

pre = list(input())
ans = 0
for i in range(1,h):
    s = list(input())
    for j in range(w-1):
        if (pre[j] == s[j] and pre[j+1] == s[j+1]) or (pre[j] == pre[j+1] and s[j] == s[j+1]):
            continue
        else:
            # print(pre[j:j+2])
            # print(s[j:j+2])
            ans += 1
    pre = s

print(ans)

# %% D

import math

X,Y,R = map(float,input().split())
X *= round(10**4)
Y *= round(10**4)
R *= round(10**4)


minx = math.ceil(X-R)
maxx = math.floor(X+R)
ans = 0

for x in range(minx, maxx+1):
    y= math.sqrt(R**2-(x-X)**2)
    # print(x,y)
    miny = math.ceil((Y-y)/(10**4))
    maxy = math.floor((y+Y)/(10**4))
    # print(x,miny,maxy)
    ans += maxy-miny+1

print(ans)