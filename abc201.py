# %% A

# a = list(map(int,input().split()))

# if sum(a) / 3 in a:
#     print('Yes')
# else:
#     print('No')

# %% B

n = int(input())
st = {}
for i in range(n):
    s,t = map(str,input().split())
    st[s] = int(t)

st[max(st, key=st.get)] = -1
print(max(st, key=st.get))

# %% C

