# m,d = map(int,input().split())

# ans = 0

# for i in range(2,m+1):
#     for j in range(2,d+1):
#         if j//10 >=2 and j%10 >=2 and i == (j//10) * (j%10):
#             ans += 1

# print(ans)

# n, k = map(int, input().split())
# a = list(map(int,input().split()))

# ans = 0

# for i in range(n-1):
#     for j in range(i+1, n):
#         if a[i] > a[j]:
#             ans += 1

# ansk = 0

# for i in range(n):
#     for j in range(n):
#         if a[i] > a[j]:
#             ansk += 1

# print((ans * k + ansk * ((k-1)*k //2)) % (10**9 + 7))

# x = 1000 - int(input())

# ans = 0
# for i in [500,100,50,10,5]:
#     ans += x // i
#     x = x % i

# print(ans + x)

# import re

# n,k= map(int,(input().split()))
# b= input()
# LRn=len(re.findall('LR',b))
# RLn=len(re.findall('RL',b))

# if min(LRn,RLn)>=k:
#     ans=k*2
# elif min(LRn,RLn)<=k and max(LRn,RLn)>=k:
#     ans=k+min(LRn,RLn)
# else:
#     ans=LRn+RLn

# a= list(b)
# for i in range(len(a)-1):
#     if a[i]==a[i+1]:
#         ans +=1

# print(ans)

# s = int(input())
# n= list(map(int,(input().split())))
# ans=0
# for i in range(len(n)-1):
#     for j in range(i+2,len(n)+1):
#         # print(n[i:j]," : ",sorted(n[i:j])[::-1][1])
#         ans+=sorted(n[i:j])[::-1][1]
# print(ans)

# n= int(input())
# k= int(input())
# l=[input() for i in range(n)]
# ans=set([])
# import itertools

# for i in itertools.permutations(l,k):
#         ans.add(''.join(i))

# print(len(ans))

# s= input()
# ans='Yes'
# for i in s[::2]:
#         if i not in 'RUD':
#                 ans='No'
#                 break
# for j in s[1::2]:
#         if j not in 'LUD':
#                 ans='No'
#                 break
# print(ans)
# import collections
# n, k, q = map(int, input().split())
# a= [int(input()) for i in range(q)]
# c=collections.Counter(a)

# for i in range(n):
#         if q - c[i+1] < k:
#                 print('Yes')
#         else:
#                 print('No')

# import bisect

# import heapq
# n, m = map(int,input().split())
# a = list(map(float, input().split()))

# for i in range(m):
#         atop = a.pop()
#         bisect.insort_left(a,atop/2)

# print(sum([int(i) for i in a]))

# n= int(input())
# s = input()

# for i in range(n//2,0,-1):
#         for j in range(0,n-i*2):
#                 if s[0:i] in s[i+j+1:]:
#                         print(i)
#                         break
#         else:
#                 print(0)


# n = int(input())
# if n==1:
#         print(1.)
# elif n % 2 ==0:
#         print(1/2)
# else:
#         print( ((n-1)/2+1) / n)

# n, h = map(int,input().split())
# l = sorted(list(map(int,input().split())))

# import bisect
# print(n- bisect.bisect_left(l, h))

# n = int(input())
# l = list(map(int,input().split()))
# ans = [0]*n
# for a, i in enumerate(l):
#         ans[i-1] = a+1
# print(' '.join(map(str, ans)))

#nを素因数分解したリストを返す
def prime_decomposition(n):
  i = 2
  table = []
  while i * i <= n:
    while n % i == 0:
      n /= i
      table.append(i)
    i += 1
  if n > 1:
    table.append(n)
  return table

def divisor(n): #nの約数を全て求める
    i = 1
    table = []
    while i * i <= n:
        if n%i == 0:
            table.append(i)
            table.append(n//i)
        i += 1
    table = list(set(table))
    return table

a, b = map(int, input().split())

ans = [1]

#公約数を求める
def cf(x1,x2):
    cf=[]
    for i in range(2,min(x1,x2)+1):
        if x1 % i == 0 and x2 % i == 0:
            cf.append(i)
    return cf

cflist = cf(a,b)

print(cflist)
import fractions
import math
import itertools

# for i in at:
#         for j in bt:
#                 if 1 == math.gcd(i,j):
#                         ans += 1
# for i in cflist:
#         ans.append(i)
#         for j in itertools.combinations(ans, 2):
#                 if 1 == math.gcd(j[0],j[1]):
#                         pass# print(i)
#                 else:
#                         ans.pop()
#                         break


# print(len(ans))
print(prime_decomposition(math.gcd(a,b)))
print(len(set(prime_decomposition(math.gcd(a,b))))+1)
