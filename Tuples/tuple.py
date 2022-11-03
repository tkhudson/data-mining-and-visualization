#tuple
'''
tuple1= (2,3,5,6)
print (type(tuple1))

print (tuple1[1])

print (tuple1)

list1= list(tuple1)
print (list1)

print (tuple(list1))

tup1= (1,2,3)
tup2 = (4,5,6)
print (tup1+tup2)

a,b,c,d = tuple1
print (b)

tuple2 = (2,)
print (type(tuple2))
'''
'''
#ex2
tuple3= (1,2,7,5,4,3,'s','r',4,8)
list2= list(tuple3)
remove_items = [5,4,3]
for i in remove_items:
    list2.remove(i)
print  (list2)
print (tuple(list2))

t1= tuple3[:3]
print (t1)
t2= tuple3[6:]
print(t1+t2)

#ex4
sum = 0
a = [[2,3,4,6], [1,2], [7,7,10]]

for i in a:
    for j in i:
        sum+=j

print(sum)
'''
'''
#ex5
daily_sales = []

for i in range (3):
    a = float(input("Enter sales for the day: "))
    daily_sales.append(a)
#print( daily_sales)
print (sum(daily_sales))
max1 = max(daily_sales)
print(max1)
'''
#dictionary
a= {1:23,2:34,3:89}
print (a[3])

print (len(a))

a[4] = 100
print (a)

a[2] =45
print (a)

del a[2]
print(a)

a.clear()
print(a)

dic1 = {1:23,2:34,3:89}

for key in dic1.keys():
    print(key)

for key in dic1.values():
    print(key)
    
for key in dic1.items():
    print(key)

dic1.pop(2)
print (dic1)

dic1.popitem()
print (dic1)

dic3 = {1:14,2:56,3:96}
#ex7

dic2= {'CIS101':['xxxx',45], 'CIS103':['yyyy',35], 'CIS104':['sss',52],}
print (dic2['CIS101'])

#ex8
for key in dic1.keys():
    print (key, ';', dic3[key])

#ex9
for key in sorted(dic3.keys(), reverse = True):
    print (key, ';', dic3[key])




    
                              



