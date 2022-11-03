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

 
