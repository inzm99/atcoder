# n=int(input())
# if n % 1000 == 0:
#     print(0)
# else:
#     print(1000 - (n % 1000))

# tcase=['AC','WA','TLE','RE']
# tcount=[0,0,0,0]

# n=int(input())
# for i in range(n):
#     c=input()
#     for t in range(4):
#         if tcase[t]==c:
#             tcount[t]+=1
#             break

# for i in range(4):
#     print(tcase[i],'x',tcount[i])

# h,w,k=map(int,input().split())
# c=[[]]*h
# for i in range(h):
#     tmp =list(input())
#     c[i] = [1 if x=='#' else 0 for x in tmp]

# # print(c)
# from itertools import product

# aans=0

# for i in product([0,1],repeat=h):
#     for j in product([0,1],repeat=w):
#         ans=0
#         for hh in range(h):
#             for ww in range(w):
#                 if i[hh] == 1 and j[ww] == 1:
#                     if c[hh][ww] == 1:
#                         # print(i,j)
#                         ans += 1
#         if ans == k:
#             aans +=1

# print(aans)

n =int(input())
A=list(map(int,input().split()))
A = A+A
ans=0
A.sort()
A.pop()

for _ in range(n-1):
    ans+= A.pop()
print(ans)

