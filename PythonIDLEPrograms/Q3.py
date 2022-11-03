dic1 = {'L1': ['NY', 'CT', 'NH', 'MA'], 'L2': ['TX', 'NM'], 'L3': ['CA',   'WA', 'AZ'], 'L4': ['ND', 'SD','WY', 'ID'], 'L5':['UT'],'L6':['MN','WI','KY']} 

with open ('Q3Output.txt', 'w')as outfile:
    for key,item in dic1.items():
        d = len(item)
        print (key, ":", "Number of states-", d, file=outfile)
