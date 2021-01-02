# # %%

# a,b = map(str,input().split())

# suma = int(a[0])+int(a[1])+int(a[2])
# sumb= int(b[0])+int(b[1])+int(b[2])

# if suma >= sumb:
#     print(suma)
# else:
#     print(sumb)

# %%

# n = int(input())
# a= [0]*n
# b=[0]*n
# ans = 0

# for i in range(n):
#     b[i],a[i] = map(int,input().split())

# for i in range(n):
#     for j in range(i+1,n):
#         if b[i] == b[j]:
#             continue
#         elif -1 <= (a[i]-a[j]) / (b[i]-b[j]) <= 1:
#             ans += 1

# print(ans)

# %%

# n = int(input())
# groupa =set()

# groupb =set()

# for i in range(n):
#     a = input()
#     if a[0] == '!':
#         groupa.add(a[1:])
#     else:
#         groupb.add(a)

# intersections = list(groupa.intersection(groupb))
# if intersections:
#     print(intersections[0])
# else:
#     print('satisfiable')

# %%
import numpy as np
from bisect import bisect_left

n = int(input())
a=[0]*n
b=[0]*n
rem = [0]*n
ans = 0
for i in range(n):
    a[i], b[i] = map(int,input().split())
    rem[i] = a[i]+b[i]+a[i]

# suma = sum(a)
sumhalf = sum(a)
rem.sort(reverse=True)
for i in range(n):
    sumhalf -= rem[i]
    ans += 1
    if sumhalf < 0:
        break
print(ans)




