# %%

N, X, T = map(int,input().split())

if N % X == 0:
    print((N//X)*T)
else:
    print((N//X + 1)*T)

# %%

N = list(input())
N = [int(x) for x in N]

print('Yes') if sum(N) % 9 == 0 else print('No')

# %%

N = int(input())
A = list(map(int,input().split()))

ans = 0

for i in range(0,N-1):
    if A[i] > A[i+1]:
        ans += A[i] - A[i+1]
        A[i+1] = A[i]

print(ans)


# %%

H, W = map(int,input().split())
Ch,Cw = map(int,input().split())
Dh,Dw = map(int,input().split())

S = [[0]*W]*H

for i in range(H):
    S[i] = list(input())

print(S)
walk = [(Ch-1,Cw-1)]
group = 0
while walk:
    a,b = walk.pop()

    if a == Dh and b == Dw:
        ans = walk[a][b]
        break

    for x,y in [(a-1,b),(a+1,b),(a,b+1),(a,b-1)]:
        if x < 0 or x > H or y < 0 or y > W:
            continue
        if walk[x][y] == '.':
            walk[x][y] = group
            walk.append((x,y))
        elif  walk[x][y] == group or walk[x][y] == '#':
            continue
        else:
            walk[x][y] = max(walk[x][y], group)
    
    for x, y in [(a-2,)]:
        if x < 0 or x > H or y < 0 or y > W:
            continue
        if S[x][y] == '.':
            S[x][y] = group + 1
            walk.append((x,y))

print(S[Dh][Dw])
# %%
