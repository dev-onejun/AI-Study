import codecs

filename = 'list-euckr.csv'
csv = codecs.open(filename, 'r', 'euc_kr').read()

data = []
rows = csv.split('\r\n')
for row in rows:
    if row == '': continue
    cells = row.split(",")
    print(cells)
    data.append(cells)


for c in data:
    print(c[1], c[2])
