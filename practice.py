# # C - Many Formulas
# S = input()
# N = len(S)
# ans = 0

# for i in range(1,2**(N-1)):
#     A = S[0]
#     for j in range(N-1):
#         if 1 << j & i:
#             A += '+'
#         A += S[j+1]
#     ans += eval(A)
#     # print(A, 'ans: ', ans)

# print(ans+int(S))

# # C - Train Tickt

# ABCD = input()

# for i in range(1,2**3):
#     X = ABCD[0]
#     for j in range(3):
#         if 1 << j & i:
#             X += '+'
#         else:
#             X += '-'
#         X += ABCD[j+1]
#     # print('eval: ',X)
#     if eval(X) == 7:
#         print(X+'=7')
#         break

# # C - All Green

# D, G = map(int, input().split())
# p = [0]*D
# c = [0]*D

# question = 0

# for i in range(D):
#     p[i], c[i] = map(int, input().split())

# question = []

# for x in range(2**D):
#     # print(bin(x))
#     questioni = 0
#     not_comp = []
#     score = 0
#     for y in range(D):
#         # print( bin(1<<y&x) )
#         if 1 << y & x:
#             score += c[y] + p[y]*100*(y+1)
#             questioni += p[y]
#             print('Mscore: ',score,'questioni: ',questioni)
#         else:
#             not_comp.append(y)
#     for y in not_comp[::-1]:
#         for z in range(p[y]-1):
#             if score < G:
#                 score += 100*(y+1)
#                 questioni += 1
#                 print('score: ',score,'questioni: ',questioni)
#             else:
#                 break
#         # else:
#         #     continue
#         # break
#     if score >= G:
#         question.append(questioni)

# print(min(question))

# A - 高橋君とお肉

# N = int(input())
# t=[]
# for i in range(N):
#     t.append(int(input()))

# ans = []
# for i in range(2**N):
#     a=[]
#     b=[]
#     for j in range(N):
#         if 1 << j & i:
#             a.append(t[j])
#         else:
#             b.append(t[j])
#     ans.append(max(sum(a),sum(b)))

# print(min(ans))

# # D - 派閥

# N, M = map(int, input().split())
# # G = []

# # for i in range(0,N-1):
# #     for j in range(i+1, N):
# #         G.append( 1 << i | 1 << j )
# # print(G)

# H = []

# for i in range(M):
#     x, y = map(int, input().split())
#     H.append(1 << (x-1) | 1 << (y-1))

# # print(H)

# ans = 1

# for j in range(2**N):
#     member = []
#     for k in range(N):
#         if 1 << k & j:
#             member.append(k+1)
#     for m in range(len(member)):
#         for l in range(m+1, len(member)):
#             if 1 << (member[m]-1) | 1 << (member[l]-1) in H:
#                 pass
#             else:
#                 break
#         else: # break で抜けなかったときに実行
#             continue
#         break # break で抜けたときに実行
#     else: # break で抜けなかったときに実行
#         ans = max(ans, len(member))

# print(ans)

################ DFS 深さ優先探索 #####################

# A - 深さ優先探索

# s = list(input())
# t = int(input())
# answer=0

# def classify(s):
#     ans=0
#     for i in range(len(s)-1):
#         if s[i] == s[i+1]:
#             s[i+1] = '0'
#             ans +=1
#     return ans

# if t == 1:
#     print(classify(s))
# else:
#     s2 = s*2
#     s21 = s*2 + list(s[0])
#     t2 = t//2
#     t2a = t%2
#     c1 = classify(s)
#     c2 = classify(s2)
#     c21 = classify(s21)
    
#     if c1*2 == c2:
#         answer = c1 * t
#     elif c2 == c21:
#         #iii type
#         answer = c2 * t2 + c1 * t2a
#     else:
#         answer = c1 * t + t - 1
         
#     print(answer)

# n = int(input())
# x = int(input())

# k = 4

# def searchc(k):
#     first = k
#     count = 0
#     temp = []
#     for i in range(100):
#         count += 1
#         if k%2 == 0:
#             k = k//2 + 2**(n-1)
#         else:
#             k = k//2
#         print(k)
#         temp.append(k)
#         if k == first:
#             ccount = {i:count for i in temp}
#             print(ccount)
#             break
#     return ccount


# a,b = map(int, input().split())

# print( a-2*b if a-2*b > 0 else 0 )
 
# a * b + b*c * c*a

# n = int(input())
# b = list(map(int,input().split()))

# ans = 0
# for i in range(n-1):
#     ans += b[i] * sum(b[i+1:])

# print(ans)

# import itertools
# import bisect

# n= int(input())
# l = list(map(int,input().split()))
# ans = 0
# L = sorted(l)

# for i in range(n-2): # max
#     for j in range(i+1, n-1): # second max
#         index = bisect.bisect_left(L[j+1:], L[i]+L[j])
#         ans += index
    
# print(ans)

# import collections

# n = int(input())
# l = list(map(int,input().split()))
# dic = collections.Counter(l)
# val = sorted(dic.values())

# print(n) # 1
# for k in range(2, n+1):
#     if len(val) < k:
#         print(0)
#     else:
#         print(sum([min(k ,i) for i in val]) // k)

# for i in range(2,n+1): # select k
#     tmp = 0
#     if len(val) < i:
#         print(0)
#     else:
#         for j in range(len(val) - i + 1):
#             print(val)
#             tmp += val[j]
#             for x in range(1,i):
#                 val[x] -= val[j]
#     print(tmp)

# from math import gcd
# from functools import reduce

# def gcdd(*numbers):
#     return reduce(gcd, numbers)
# # from fractions import gcd
# from functools import reduce

# def gcdd(*numbers):
#     return reduce(gcd, numbers)

# n, k = map(int,input().split())
# a = list(map(int,input().split()))

# agcd = gcdd(*a)

# if max(a) > k and k%agcd ==0:
#   print('POSSIBLE')
# else:
#   print('IMPOSSIBLE')

# n, m, l = map(int,input().split())
# A = [[float("INF") for i in range(n+1)] for j in range(n+1)]
# for i in range(n+1):
#     A[i][i] = 0 #自身のところに行くコストは０
# for i in range(m):
#     a,b,c= map(int,input().split())
#     if c <= l:
#         A[a][b] = c
#         A[b][a] = c
# q= int(input())
# s= [0]*q
# t= [0]*q
# for i in range(q):
#     s_,t_ = map(int,input().split())
#     s[i]=s_
#     t[i]=t_


# def warshall_floyd(d,n):
#     #d[i][j]: iからjへの最短距離
#     for k in range(n):
#         for i in range(n):
#             for j in range(n):
#                 d[i][j] = min(d[i][j], d[i][k] + d[k][j])
#     return d

# mincost = warshall_floyd(A,n+1)

# for i in range(q):
#     if mincost[s[i]][t[i]] == float("INF"):
#         print(-1)
#     else:
#         print(mincost[s[i]][t[i]] // l) # maybe not correct


# a,b = map(int,input().split())
# print(a*b if max(a,b)<10 else -1) 

# kuku =[]

# for i in range(1,10):
#     for j in range(1,10):
#         kuku.append(i*j)

# print('YES' if int(input()) in kuku else 'NO')

# def factorization(n):
#     temp = n
#     for i in range(int(-(-n**0.5//1)),0,-1 ):
#         if temp%i==0:
#             b = temp//i
#             return i,b

# a,b = factorization(int(input()))
# print(a-1+b-1)

# import math

# a, b, x = map(int,input().split())

# if a*a*b/2 <= x:
#     h = (a*a*b - x)*2 / (a*a)
#     print(math.degrees(math.atan(h/a)))
# else:
#     h = (2*x) / (a*b)
#     print(math.degrees(math.atan(b/h)))

# import collections

# n = int(input())

# d = list(map(int,input().split()))

# waru = 998244353

# c = collections.Counter(d)

# ans =1

# cmax =max(c.keys())
# if d[0] != 0:
#     print(0)
# else:
#     for i in range(1,cmax+1):
#         if d[i] == 0:
#             ans = 0
#             break
#         ans *= c[i-1]** c[i]

#     print(ans % waru)

# 1            1
# 2 2 2
#            222
  # １の数
# 22         2 
# 2          22
  # 
# 3＊(1 + 1 + 3 + 3 )

# import numpy as np

# n = int(input())
# a = np.array(list(map(int,input().split())))
# b = np.array(list(map(int,input().split())))
# a_sort = np.sort(a)
# b_sort = np.sort(b)

# c = b - a
# np.where()
# print(- a + b)
# print(min(- a_sort + b_sort))

# n = int(input())
# s = input()

# if n%2 == 1:
#   print('No')
# elif s[:n//2] == s[n//2:]:
#   print('Yes')
# else:
#   print('No')


# import itertools
# import math

# n = int(input())
# s = []

# for i in range(n):
#     s.append(list(map(int,input().split())))

# def distance(a,b):
#     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# tmp =0
# # comb = len(itertools.combinations(s,2))
# for j in itertools.combinations(s,2):
#     tmp += distance(j[0],j[1])

# print(tmp*2/n)

import math

# x, y = map(int,input().split())

# def cmb(n, r, mod):
#     if ( r<0 or r>n ):
#         return 0
#     r = min(r, n-r)
#     return g1[n] * g2[r] * g2[n-r] % mod

mod = 10**9+7 #出力の制限
# N = 10**4
# g1 = [1, 1] # 元テーブル
# g2 = [1, 1] #逆元テーブル
# inverse = [0, 1] #逆元テーブル計算用テーブル

# def cmb(n, k,MOD):
#     if k > n - k:
#         k = n - k
#     ans = 1
#     for i in range(1, k + 1):
#         ans *= n + 1 - i
#         ans *= pow(i, MOD - 2, MOD)
#         ans %= MOD
#     return ans

# # a = cmb(n, r)

# # for i in range( 2, N + 1 ):
# #     g1.append( ( g1[-1] * i ) % mod )
# #     inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
# #     g2.append( (g2[-1] * inverse[-1]) % mod )

# if (x+y)%3 != 0:
#     ans = 0
# else:
#     m = (2*x - y) //3
#     n = x - 2*m
#     # print(m,n)
#     if m <0 or n < 0 :
#         ans = 0
#     else:
#         ans = cmb(n+m, m, mod) #math.factorial(n+m) // (math.factorial(n+m - m) * math.factorial(m))

# print(ans)

# def combinations_count(n, r):
#     # 組み合わせの総数　c = n! / (r! * (n - r)!)
#     return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

# import itertools

# array=[1,3,5,7,9]
# cumsum = itertools.accumulate(array)
# print(list(cumsum)) # [1,4,9,16,25]
# s = ['ab','bc','cd']
# print(list(itertools.accumulate(s))) # ['ab','abbc','abbccd']
# bi = [0,0,0,1,1,0,0,0,1,1,0,1]
# gr = itertools.groupby(bi)
# for key, group in gr:
#     print(f'{key}: {list(group)}')
# # 0: [0, 0, 0]
# # 1: [1, 1]
# # 0: [0, 0, 0]
# # 1: [1, 1]
# # 0: [0]
# # 1: [1]
# itertools.permutations([1,2,3]) # -> [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
# # 順列組み合わせ
# itertools.combinations([1,2,3],2) # [(1,2),(2,3),(1,3)]
# # 組み合わせ
# itertools.product([0,1],repeat=3) # -> [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
# # 直積 (x,y) for x in X for y in Y


# n, a, b = map(int,input().split())

# if abs(a-b) % 2 == 0:
#     print(abs(a-b)//2)
# else:
#     print(min(n - min(a,b), max(a,b)-1, min(a,b)-1+abs(a-b)//2+1, n-max(a,b)+abs(a-b)//2+1))

# n, m, v, p = map(int,input().split())
# a = sorted(list(map(int,input().split())))
# b= [ai+m for ai in a ]
# print(a)
# print(b[:v])
# for i in range(n):
#     # iが選ばれる場合
#     a[i] + m
#     sum(a[i+1:]) + m * (v - i + 1) // (n- i) 

#     ans = n-i
# print(ans)

# a,b,k = map(int,input().split())

# if a>k:
#     print(a-k, end=" ")
#     print(b)
# elif a+b >= k:
#     print(0, end=" ")
#     print(a+b-k)
# else:
#     print("0 0")

# import math

# def is_prime(n):
#     if n == 1: return False

#     for k in range(2, int(math.sqrt(n)) + 1):
#         if n % k == 0:
#             return False

#     return True

# x = int(input())
# for i in range(10**5):
#     if is_prime(x+i):
#         print(x+i)
#         break

# import re

# n, k = map(int,input().split()) # K 回前のじゃんけんで出した手と同じ手を出すことはできない
# r, s, p = map(int,input().split()) # r:goo , s:choki, p:paa
# t = input()

# # aaaaa
# ans = 0

# for i in range(k):
#     tmp = t[i::k]
#     tmp = re.sub('rr','rz',tmp)
#     tmp = re.sub('ss','sz',tmp)
#     tmp = re.sub('pp','pz',tmp)
#     ans += tmp.count('r') * p + tmp.count('s') * r + tmp.count('p') * s
#     # print(tmp)
# print(ans)


# import itertools

# n, m = map(int,input().split())
# a = list(map(int, input().split()))
# a = sorted(a,reverse=True)

# ans =0
# x = m // 3
# y = m % 3

# for a, b in itertools.combinations_with_replacement(a,2):
#     print(a,b)
#     if m >= 3:
#         ans += a*4 + b*2
#         m-= 3
#     elif m == 2:
#         ans += a*3 + b*1
#         break
#     elif m == 1:
#         ans += a*2
#         # b
#         break
#     elif m == 0:
#         break
# # if y == 0:
# #     ans  += a[0] *4 + sum(a[1:x]) * 6 + a[x] * 2
# # elif y == 1:
# #     ans  += a[0] *4 + sum(a[1:x]) * 6 + a[x] * 4
# # elif y == 2:
# #     ans  += a[0] *4 + sum(a[1:x]) * 6 + a[x] * 5
# print(ans)


# import numpy as np
 
# N,M = map(int,input().split())
# A = np.array(list(map(int,input().split())))
# A.sort()
 
# def shake_cnt(x):
#     X = np.searchsorted(A,x-A)  #numpy内蔵のbisect_left, np.array一括で検索可
#     print(X)
#     return N*N - X.sum()
#     #行わない握手のcnt, X.sum()は行う握手のcnt ? 逆か?
 
# #にぶたん
# L = 0
# R = 10**6   #範囲外
# while L+1 < R:
#     mid = (L+R) // 2
#     if shake_cnt(mid) >= M:
#         L = mid
#     else:
#         R = mid
# #増える幸福度がL以上になる握手の方法がM通り ?
 
# Acum = np.zeros(N+1)
# Acum[1:] = np.cumsum(A) #この書き方を覚えておく、np.arrayで先頭に0を残せる
# X = np.searchsorted(A,R-A)
# shake = N*N - X.sum()
# ans = (Acum[-1] - Acum[X]).sum() + (A*(N-X)).sum()
# #Acum[-1]は定数, Acum[X]はnp.array A,Xはnp.array
# ans += (M-shake)*L
# print(int(ans))

# n, q = map(int,input().split())

# # ans = 2**n
# # for i in range(q):
# #     l, r = map(int,input().split())
# #     ans ^= 2**(n-l+1) - 1
# #     ans ^= 2**(n-r) - 1

# # print(bin(ans)[3:])

# o = [0] * (n+1)

# for i in range(q):
#     l, r = map(int,input().split())
#     o[l-1] += 1
#     o[r] += 1

# for i in range(n):
#     if i > 0:
#         o[i] += o[i-1]
#     print(o[i]%2 , end="")
# print()

# n = int(input())
# o = [0] * 1000002

# for i in range(n):
#     l,r = map(int,input().split())
#     o[l] += 1
#     o[r+1] -= 1
# for i in range(n):
#     o[i+1] += o[i]
# print(max(o))

# k,x = map(int,input().split())
# print('Yes' if k*500 >= x else 'No')

# import math
# import bisect
# n= int(input())
# a=list(map(int,input().split()))
# b=list(map(int,input().split()))
# A=0
# B=0
# for i in range(n-1):
#     an=bisect.bisect_left(sorted(a[i:]),a[i])
#     # print('an:',an)
#     A+= an *math.factorial(n-i-1)
#     # print(A)
#     bn=bisect.bisect_left(sorted(b[i:]),b[i])

#     B+= bn *math.factorial(n-i-1)
#     # print(B)

# print(abs(A-B))

# x,y= map(int,input().split())

# a= str(x)*y
# b= str(y)*x

# print(a if a<b else b)

# n = int(input())
# p = list(map(int,input().split()))

# ans = 0
# minp = p[0]

# for i in range(n):
#     if minp >= p[i]:
#         ans += 1
#         minp = p[i]

# print(ans)
    
# n=int(input())
# if n<10:
#     print(n)
# else:
#     startc= str(n)[0]
#     endc=str(n)[-1]
#     lenc = len(str(n))
#     c1 = 9
#     csame = 9
#     for i in range(1,lenc):
#         c1+= 9**(i-1)*8
#         csame += 9**(i-2)
#     if int(startc) >= int(endc):
#         c1+= 


# sxxxxx1
# sxxxxxe

# from math import gcd
# from functools import reduce

# def gcdd(numbers):
#     return reduce(gcd, numbers)

# def lcm_base(x, y):
#     return (x * y) // gcd(x, y)

# def lcm(*numbers):
#     return reduce(lcm_base, numbers, 1)

# def lcm_list(numbers):
#     return reduce(lcm_base, numbers, 1)

# n=int(input())
# a=list(map(int,input().split()))
# # import numpy as np
# lcmn = lcm_list(a)
# ans = 0
# for i in range(n):
#     ans += (lcmn // a[i])  
# print(ans% (10**9+7))

# s,t= map(str,input().split())
# a,b=map(int,input().split())
# u=input()

# if s==u:
#     print(a-1,b)
# else:
#     print(a,b-1)

# print('x'* len(input()))

# n=int(input())
# a=list(map(int,input().split()))
# a=set(a)
# if n==len(a):
#     print('YES')
# else:
#     print('NO')

# n,k=map(int,input().split())
# p=list(map(int,input().split()))
# ans= 0
# tmp=0
# sump=sum(p[0:k])
# for i in range(n-k):
#     sump=sump+p[i+k]-p[i]
#     # print(i,tmp,p[i:i+k])
#     if sump>tmp:
#         # print(i,tmp)
#         tmp = sump
#         ans= i+1
# anss=0
# for j in p[ans:ans+k]:
#     anss+=(j+1)/2


# print(anss)

# n=input()
# nn = int(nn)
# lenn=len(n)
# K=int(input())

# dp[i][j][k]
# i桁目まで決めたとき、j個の０でないものを使って
# k=0：そこまでの桁はNと一致
# k=1：そこまでの桁ですでにN以下であることが確定

# EDPC

##### keta DP ######

# import sys
# import math
# # sys.setrecursionlimit(10 ** 7)
# from functools import lru_cache

# N=int(input())
# K =int(input()) 

# @lru_cache(None)
# def dp(N, K):
#   if N < 10:
#     if K == 0:
#       return 1
#     if K == 1:
#       return N
#     return 0
#   q, r = divmod(N, 10)
#   ret = 0
#   if K >= 1: # 下一桁が0でないときはKを1消化
#     ret += dp(q, K-1) * r # 下一桁がr以下の時
#     ret += dp(q-1, K-1) * (9-r) # 下一桁がrより上の時はq-1であれば成立
#   ret += dp(q, K) # 下一桁が0のとき
#   return ret

# print(dp(N, K))


# a,b,c = map(int,input().split())
# if a==b or b==c or a==c:
#   if a==b==c:
#     print('No')
#   else:
#     print('Yes')
# else:
#   print('No')
  
# n= int(input())
# a = list(map(int,input().split()))

# ans = 'APPROVED'
# for i in a:
#   if i%2 == 0 and i%3 != 0 and i%5 != 0:
#       ans = 'DENIED'
#       break

# print(ans)

# n= int(input())
# from collections import Counter
# S = []
# for _ in range(n):
#   s = input()
#   S.append(s)

# c= Counter(S)
# ans =[]
# maxvalue =0
# # print(c.most_common())
# for values, counts in c.most_common():
#   if maxvalue > counts:
#     break
#   else:
#     maxvalue = counts
#   ans.append(values)
#   # print(values)
  
# for i in sorted(ans):
#   print(i)
# # print(sorted(ans))


# n, k = map(int,input().split())
# a = list(map(int,input().split()))

# over0 = [i for i in a if i>0]
# under0 = [i for i in a if i<0]
# numover0 = len(over0)
# numunder0 = len(under0)
# num0 = len(a) - numover0 - numunder0
# l=[]
# if k <= numover0 * numunder0: # under 0
#   for o in over0:
#     for u in under0:
#       l.append(o*u)
#   print(sorted(l)[k-1])
#   #under 0
# elif k<= numover0 * numunder0 + num0 * (numover0+numunder0): # 0
#   print(0)
# else: # over 0
#   #over 0
#   for i in range(numover0-1):
#     for j in range(i+1,numover0):
#       l.append(over0[i]*over0[j])
#   for i in range(numunder0-1):
#     for j in range(i+1,numunder0):
#       l.append(under0[i]*under0[j])
#   print(sorted(l,reverse=True)[len(a)*(len(a)-1)//2-k])


# n,r=map(int,input().split())

# for i in range(100):
#   if n<r**i:
#     print(i)
#     break

# n=int(input())
# import numpy as np
# npi = np.array(list(map(int,input().split())))
# if npi.mean()%1>0.5:
#   p=npi.mean()//1 +1
# else:
#   p=npi.mean()//1
# print(int(np.sum((npi-p)**2)))

# import math
# def combinations_count(n, r):
#   print(n,r)
#   return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

# def cmb(n, r, mod):
#     if ( r<0 or r>n ):
#         return 0
#     r = min(r, n-r)
#     return g1[n] * g2[r] * g2[n-r] % mod

# mod = 10**9+7 #出力の制限
# N = 10**9
# g1 = [1, 1] # 元テーブル
# g2 = [1, 1] #逆元テーブル
# inverse = [0, 1] #逆元テーブル計算用テーブル

# for i in range( 2, N + 1 ):
#     g1.append( ( g1[-1] * i ) % mod )
#     inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
#     g2.append( (g2[-1] * inverse[-1]) % mod )


# waru = 10**9+7

# a = cmb(4,2,waru)
# print(a)

# n,a,b=map(int,input().split())
# ans=0
# for i in range(1,n+1):
#   if i not in [a,b]:
#     ans += cmb(n,i,waru)
#     # ans = ans % waru
# print(ans)

# a,b=map(int,input().split())

# n = int((a/0.08)//1)
# ans = -1

# for i in range(n,n+10**2):
#   if int((i*0.08)//1) == a and int((i*0.1)//1) == b:
#     ans = i
#     break
#   else:
#     continue

# print(ans)

# import sys
# input = sys.stdin.readline
# # print(s)
# s = list(input().rstrip())
# q = int(input()) #int(input())
# inv = 1
# query = [input().rstrip() for _ in range(q)]
# bef = []
# aft = []


# for t in query:
#   if t == '1':
#     inv *= -1 #s = s[::-1]
#   else:
#     _, f, c = map(str, t.split())
#     if (f =='1' and inv == 1) or (f =='2' and inv == -1):
#       # s = c+s
#       bef.append(c)
#     else:
#       # s = s+c
#       aft.append(c)

# bef.reverse()
# bef.extend(s)
# bef.extend(aft)

# print(''.join(bef[::inv]))

# """
# a
# 6
# 2 2 a
# 2 1 b
# 1
# 2 2 c
# 1
# 1
# """

# x,y = map(int,input().split())
# a = sorted(list(map(int,input().split())))

# hikaku = sum(a)/(4*m)

# if a[-m] >= hikaku:
#   print('Yes')
# else:
#   print('No')


# import heapq
# import sys

# a=[1,2,3,4,5,6,7,8,9]
# heapq.heapify(a)

# k = int(input())

# if k<10:
#   print(k)
#   sys.exit()

# ans = 9

# while True:
#   x = heapq.heappop(a)
#   # print(a)
#   if x%10 ==0:
#     for i in [0,1]:
#       y = x*10 + i
#       ans += 1
#       if ans == k:
#         print(y)
#         sys.exit()
#       heapq.heappush(a,y)
#   elif x%10 == 9:
#     for i in [-1,0]:
#       y = x*10 + x%10 + i
#       ans += 1
#       if ans == k:
#         print(y)
#         sys.exit()
#       heapq.heappush(a,y)
#   else:
#     for i in [-1,0,1]:
#       # print(i)
#       y = x*10 + x%10 + i
#       ans += 1
#       if ans == k:
#         print(y)
#         sys.exit()
#       heapq.heappush(a,y)

# n,k,c = map(int,input().split())
# S = input()

# left = [ -1 for i in range(k)]
# right = [ -1 for i in range(k)]

# pointer = 0
# for l in range(k):

#   for s in range(pointer,n):
#     if S[s] == 'o':
#       left[l] = s
#       pointer = s+c+1
#       break

# pointer = n-1
# for r in range(k):

#   for s in range(pointer,-1,-1):
#     if S[s] == 'o':
#       right[r] = s
#       pointer = s-c-1
#       break

# for l,r in zip(left, right[::-1]):
#   if l==r:
#     print(l+1)

# %%

# x = input()
# azlist = list('abcdefghijklmnopqrstuvwxyz')
# if x in azlist:
#   print('a')
# else:
#   print('A')

# %%

# n= int(input())

# azlist = list('abcdefghijklmnopqrstuvwxyz')
# ans = ''
# for i in range(1,10000):
#   n -= 1
#   ans = azlist[n%26] + ans
#   n = n//26
#   if n ==0:
#     break

# s=input()
# t=input()
# ans=0
# n = len(s)
# for i in range(n):
#   if s[i]!= t[i]:
#     ans+= 1

# print(ans)

n,m,k=map(int,input().split())
a=list(map(int,input().split()))
b=list(map(int,input().split()))
ans=0
aindex=n
bindex=m
acumsum = [0]
bcumsum = [0]

ab =0
for aa in range(n):
  ab += a[aa]
  acumsum.append(ab)# = ab
  if ab>k:
    aindex= aa
    break

ab =0
for bb in range(m):
  ab += b[bb]
  bcumsum.append(ab)
  if ab>k:
    bindex= bb
    break

j= bindex
for i in range(aindex+1):
  if acumsum[i] >k:
    break
  while bcumsum[j] > k-acumsum[i]:
    j -=1
  ans = max(ans, i+j)

print(ans)




# ab= sum(a) + sum(b)

# while(ab > k):
#   print(ab,a,b)
#   if len(a)==0:
#     ab -= b.pop(-1)
#     ans -=1
#   elif len(b)==0:
#     ab -= a.pop(-1)
#     ans -=1    
#   elif a[-1] >= b[-1]:
#     ab -= a.pop(-1)
#     ans -=1
#   else:
#     ab -= b.pop(-1)
#     ans -=1
# print(ans)

