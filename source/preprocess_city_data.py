with open("city.txt", "r", encoding="utf-8") as f:
    data = f.readlines()
    cities = []
    for i in data:
        cities.append(i.strip() + '@地址')
        if i.strip()[-1] == '市':
            cities.append(i.strip()[:-1] + '@地址')
print(len(cities),cities[:10])
with open('prediction_city.txt', 'w', encoding='utf-8') as f:
    for i in cities:
        f.write(i + '\n')


