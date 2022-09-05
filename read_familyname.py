with open ("familyname.txt",'r',encoding='utf-8') as f:
    data = f.readlines()
    names = []
    for i in data:
        for j in i.split():
            names.append(j[-1])

print(len(names))
print(names[80:])
