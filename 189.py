# %% A

# C = input()

# if C[0] == C[1] and C[1] == C[2]:
#     print('Won')
# else:
#     print('Lost')

# %% B

# N, X = map(int,input().split())

# VP = []
# X *= 100

# for i in range(N):
#     VP.append(tuple(map(int,input().split())))

# for j in range(N):
#     v,p = VP[j]
#     X -= v * p
#     if X < 0:
#         print(j+1)
#         break
# else:
#     print(-1)

# %% C
N = int(input())
A = list(map(int,input().split()))

ans = 0#N * min(A)

# sortedA = sorted(A)
for left in range(N):
    x = A[left]
    for right in range(left,N):
        x= min(x, A[right])
        ans = max(ans, x*(right - left +1))

print(ans)


# %% D

