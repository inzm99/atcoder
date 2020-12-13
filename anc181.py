# # %% A

# if int(input()) % 2 ==0:
#     print('White')
# else:
#     print('Black')

# #%% B

# n=int(input())
# ans=0

# for i in range(n):
#     a,b=map(int,input().split())
#     ans += (a+b)*(b-a+1)//2
# print(ans)

# %% C

import sys
def axy(x1,y1,x2,y2):
    if x1==x2:
        print('error')
        return False
    a=(y1-y2)/(x1-x2)
    b=(y1+y2-a*(x1+x2)) /2
    return a,b

n=int(input())
xy=[]
xx = [0] * 1000
# yy = [0] * 1000
ans = 'No'

for i in range(n):
    a,b=map(int,input().split())
    xy.append([a,b])
    xx[a]+=1
    # yy[b]+=1
    if xx[a]>=3:# or yy[b]>=3:
        ans='Yes'

if ans =='No':
    for i in range(n):
        for j in range(i):
            if xy[i][0] == xy[j][0]:
                continue
            a,b = axy(xy[i][0],xy[i][1],xy[j][0],xy[j][1])
            for k in range(j):
                if a*xy[k][0] + b == xy[k][1]:
                    ans='Yes'
                    break

print(ans)

#%% D

# for i in range(100,150):
#     print(8*i)

s=list(input())
s.sort()
lens= len(s)
S = int('8'*lens)

for i in range(10**(lens-1)//8-1, 10**lens // 8 +1):

    if sorted(list(str(i*8))) == s:
        print('Yes')
        break
else:
    print('No')


# while( S < 10**(lens-1)):
#     if sorted(list(str(S))) == s:
#         print('Yes')
#         break
#     S+=8
# print('No')