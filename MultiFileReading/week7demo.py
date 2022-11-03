#ex1
'''
sum = 0
count = 0
with open ("Numbers.txt", "r") as infile:
    for line in infile:
        count+=1
        sum= sum+int(line)
    average = sum/count
    print (average)
'''
'''
#ex2+3
import re, os, csv, sys
sum = 0
path = r"C:\Users\tyler\OneDrive\Documents\school2021\prog\Week 7\Ex2"
outfile = open ('file_cumulative_sum.csv', 'w', newline = '')
writer = csv.writer (outfile)
writer.writerow(['File Name', 'Sum'])
in_files = os.listdir(path)
for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    for line in text:
        sum+=int(line)
    #print (file + " : " + str(sum))
    writer.writerow([file, sum])
outfile.close()
'''
'''
#ex4a
import re, os
path = r"C:\Users\tyler\OneDrive\Documents\school2021\prog\Week 7\Ex4"

print (['File Name', 'Count'])
in_files = os.listdir(path)

for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    count = 0
    for line in text:
        if 'CIS3389' in line:
            count +=1
    print (file, str(count))
'''
#ex4b
import re, os, csv
path = r"C:\Users\tyler\OneDrive\Documents\school2021\prog\Week 7\Ex4"

print (['File Name', 'Count'])
in_files = os.listdir(path)
'''
for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    for line in text:
        if 'python' in line.lower():
            print (file, ",", line.replace('\n',''))
'''
for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    text1 = text.read()

    for sent in text1.split('.'):
        if 'python' in sent.lower():
            print (file, sent.replace('\n',''))
