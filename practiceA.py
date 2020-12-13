# -*- coding: utf-8 -*-

N = int(input())
s = list(input().split())

kekka = 0
hantei = 0

while N == sum(int(i)%2+1 for i in s):
    hantei += 1
    s = list(int(i)/2 for i in s)


print(hantei)