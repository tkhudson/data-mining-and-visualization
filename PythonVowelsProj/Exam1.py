#1

import re, os, csv, sys
sum = 0
count = 0
outfile = open ('Q1Output.csv', 'w', newline = '')
writer = csv.writer (outfile)
writer.writerow(['FileName', 'EachFileAverage'])
in_files = os.listdir(path)
with open ("file1.txt", "r") as infile:
    for line in infile:
        count+=1
        sum= sum+int(line)
    average = sum/count
    writer.writerow([file, average])
outfile.close()

for file in in_files:
    file1 = os.path.join(path, file)
    text = open(file1, "r")
    for line in text:
        count+=1
        sum= sum+int(line)
    average = sum/count


with open ("file1.txt", "r") as infile:
    for line in infile:
        count+=1
        sum= sum+int(line)
    average = sum/count

#2
exam_q2_str = """CIS3389 exam is on Monday, not on Wednesday."""

new_str = ''.join((s if s in 'aeiou' else ' ') for s in exam_q2_str)
b = new_str.split()
vowels = []
for i in b:
    vowels.append(i)

a = 0
e = 0
eye = 0
o = 0
u = 0
for i in vowels:
    if i == 'a':
        a = a + 1
    elif i == 'e':
        e = e +1
    elif i == 'i':
        eye = eye +1
    elif i == 'o':
        o = o +1
    elif i == 'u':
        u = u +1
print("a: ", a)
print("e: ", e)
print("i: ",eye)
print("o: ",o)
print("u: ",u)
