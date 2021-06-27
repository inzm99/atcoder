# # %% A

# a = list(map(int,input().split()))
# a.sort()
# print(a[1] + a[2])

# # %% B

# a,b,c,d = map(int,input().split())

# if c*d - b <= 0:
#     print(-1)
# else:
#     print(-(-a // (c*d -b)))

# %% C

n = int(input())
range_list = []
ans = 0

for i in range(n):
    t,l,r = map(int,input().split())
    if t == 1:
        range_list.append([l,r])
    elif t == 2:
        range_list.append([l,r-0.1])
    elif t == 3:
        range_list.append([l+0.1,r])
    elif t == 4:
        range_list.append([l+0.1,r-0.1])
    else:
        print('error')

range_list.sort()

ans=0
for i in range(n-1):
    for j in range(i+1, n):
        if range_list[i][1] >= range_list[j][0]:
            ans += 1
        else:
            break

print(ans)

# %% D



