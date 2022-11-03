with open ("File1.txt", "r") as infile:
    contents = infile.read()
    print (contents)
    contents = infile.readlines()
    print (contents)
    contents1 = infile.readline()
    print (contents1)
    contents2 = infile.readline()
    print (contents2)
