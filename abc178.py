# %%
import math

def mod_cmb(n: int, k: int, p: int) -> int:
    # MOD用組み合わせ
    if n < 0 or k < 0 or n < k: return 0
    if n == 0 or k == 0: return 1
    if (k > n - k):
        return mod_cmb(n, n - k, p)
    c = d = 1
    for i in range(k):
        c *= (n - i)
        d *= (k - i)
        c %= p
        d %= p
    return c * pow(d, p - 2, p) % p

def cmb(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

MOD = 10 ** 9 + 7

N = int(input())

ans = cmb(N,2) * 2 * (10**(N-2))
ans %= MOD
print(int(ans))

# %%

a,b,c,d = map(int, input().split())

print(max(a*c,a*d,b*c,b*d))

# %%
 
S = int(input())
ans = 0

for i in range(S//3):
    remain = S - (i+1)*3


print(ans)

# %%
# C
# ベン図
mod= 10**9+7
N=int(input())
# all
A = pow(10,N,mod)
# no 0
B = pow(9,N,mod)
# no 9
C = pow(9,N,mod)
# no 0 and no 9
D = pow(8,N,mod)

ans = ((A-B) + (A - C)-(A - D))%mod
print(ans)

# %%
# D
# DP

S = int(input())
mod= 10**9+7
dp = [0]* (S+1)
dp[0] = 1

for i in range(3,S+1):
    dp[i] = dp[i-3] + dp[i-1]

print(dp[S]%mod)