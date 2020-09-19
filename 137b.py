# k, x = map(int, input().split())

# for i in range(x-k+1,x+k):
#     print(i,end=" ")

# from collections import Counter

# N = int(input())
# S = []

# for i in range(N):
#     S.append(''.join(sorted(input())))

# print( sum(v*(v-1)//2 for v in Counter(S).values()))


# from collections import Counter
# print(sum(v*(v-1)//2 for v in Counter(''.join(sorted(input())) for _ in range(int(input()))).values()))

# C - たくさんの数式 / Many Formulas

S = input()

SS = S[0] + '+' + S[1]
print(SS)

ans = int(S[0]) + int(''.join(S[1:]))

print(ans)