# %%

# n,w = map(int,input().split())

# print(n//w)

# %%

# h,w = map(int,input().split())

# aa = []
# for i in range(h):
#     a = list(map(int,input().split()))
#     aa = aa + a

# print(sum(aa) - min(aa) * len(aa))

# %%

# n = int(input())
# ans = 0
# for i in range(1,n+1):
#     # print(str(oct(n)), str(n))
#     if '7' in str((oct(i))) or '7' in str(i):
#         ans += 1
# print(n-ans)

# %%

# n = int(input())
# a = list(map(int,input().split()))
# a.sort()
# suma= sum(a)
# ans = 0
# for i in range(n-1):
#     # print()
#     suma -= a[i]
#     ans += suma - a[i]*(n-i-1) 

# print(ans)

# %%

n,s,k = map(int,input().split())
# rems = set()

# for i in range(1,100000000):
#     # rem = (n*i - s) % k
#     # if rem == 0:
#     #     print((n*i - s) // k)
#     #     break
#     # if rem in rems:
#     #     print(-1)
#     #     break
#     # rems.add(rem)

#     rem = (k*i + s )% n
#     if rem == 0:
#         print(i)
#         break
#     if rem in rems:
#         print(-1)
#         break
#     rems.add(rem)
# # print(rems)

def gcd(a,b,s):
    if a==s:
        return b
    else:
        return gcd(b,a%b, s)

print(gcd(n,k,s))