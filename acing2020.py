# L,R,d = map(int,input().split())

# print(R//d - (L-1)//d)


# N=int(input())
# a=list(map(int,input().split()))

# ans=0
# for i in range(0,N,2):
#     if a[i]%2==1:
#         ans+=1

# print(ans)



# import itertools
# import math
# N=int(input())

# a = [0] * N

# if N >= 6:
#     for x,y,z in itertools.combinations_with_replacement(list(range(1,int(math.sqrt(N)) +1 )), 3):
#         f = x**2 +y**2 +z**2 + x*y + y*z + x*z
#         if f <= N:
#             if x==y==z:
#                 a[f-1] += 1
#             elif x==y or y== z:
#                 a[f-1] += 3
#             else:
#                 a[f-1] += 6
#         # elif f >= 2*N:
#         #     break
# [print(ans) for ans in a]



N = int(input())
X= input()

def popcount(n):
    a = list(str(bin(n)))
    return sum([int(i) for i in a[2:]])

waru = sum([int(x) for x in (list(X))])
X10 = int(X,2)

for i in range(N):
    ans = 1
    if X[i] == '1':
        Xe = X10 - (1 << (N-i-1))
        warue = waru-1
    else:
        Xe = X10 + (1 << (N-i-1))
        warue = waru+1
    # print(Xe,warue)
    Xe= Xe % warue
    if Xe != 0:
        for _ in range(10):
            Xe = Xe % popcount(Xe)
            ans += 1
            if Xe ==0:
                break

    print(ans)
