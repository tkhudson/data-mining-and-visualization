#1

sum = 0
print ('Number','\t','Square')
for i in range(5,31):
    if i%5==0:
        i2=i*i
        if i==10 or i==15:
            continue
    
    
        print (i,'\t',i2)
    
        sum=sum+i2
print ('Sum of squares: ', sum)

#2

a = "Sam works in a company abc in New York. He joined the company last year 2019. Before joining ABC, he used to work for a small firm in Arizona. He worked there from 2015 to 2018. Before moving to Arizona Sam used to live in South Dakota and he has been living there since 2000's."
a1 = a.lower()

year_list = []
for c in a1.split():
    c = c.strip("'s.th ,")
    
    if c.isdigit(): 
        year_list.append (c)

print (year_list)

year_count = len(year_list)
max_year = max (year_list)
min_year = min (year_list)

if 'abc' in a1:
    print (a1.count('abc'))
    
a_new =a1.replace ('h', '_')
print (a_new)

with open ("Q2Output.txt","w") as outfile:
    print ("a.", "the list of years ", year_list, file = outfile)
    print ("b.", "Count: ", year_count, ",", "Maximum: ", max_year, ",", "Minimum: ", min_year,file = outfile)
    print ("c.", "The new string: ", a_new, file = outfile)

#3

dic1 = {'L1': ['NY', 'CT', 'NH', 'MA'], 'L2': ['TX', 'NM'], 'L3': ['CA',   'WA', 'AZ'], 'L4': ['ND', 'SD','WY', 'ID'], 'L5':['UT'],'L6':['MN','WI','KY']} 

with open ('Q3Output.txt', 'w')as outfile:
    for key,item in dic1.items():
        d = len(item)
        print (key, ":", "Number of states-", d, file=outfile)

#4
        
choice = "y"
while choice.lower() == "y":
    budget = float(input("enter the budget in $: ")) 
    house_rent = float(input('Enter house rent in $: '))
    other_expenses = float(input('Enter other expenses in $: '))
    total_expenses = house_rent+other_expenses
    print ('Lina’s total budget for next month in $: ', format(budget, '.2f'))
    print ('Lina’s total expenses for next month in $:', format(total_expenses, '.2f'))
    if budget > total_expenses:
        print ("within budget")
    elif budget < total_expenses:
        print ("over budget")
    else:
        print ("matches budget") 

        
    choice = input ("Continue? y/n ")
