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

