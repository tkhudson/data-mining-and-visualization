
import re, os, csv, sys
sum = 0
count = 0
outfile = open ('Q1Output.csv', 'w', newline = '')
writer = csv.writer (outfile)
writer.writerow(['FileName', 'EachFileAverage'])
in_files = os.listdir(path)
with open ("file1.txt", "r") as infile:
for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    for line in text:
        sum+=int(line)
        #print (file + " : " + str(sum))
    writer.writerow([file, sum])
outfile.close()
'''
for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    for line in text:
        count+=1
        sum= sum+int(line)
    average = sum/count
'''
'''

with open ("file1.txt", "r") as infile:
    for line in infile:
        count+=1
        sum= sum+int(line)
    average = sum/count
'''

