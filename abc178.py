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