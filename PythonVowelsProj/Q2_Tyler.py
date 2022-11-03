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
