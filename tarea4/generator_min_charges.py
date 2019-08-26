import random

print('Ingrese nombre de archivo a crear:')
fname = input('')
print('Ingrese cantidad de particulas:')
n = int(input(''))
print('Minima posicion:')
min_pos = int(input(''))
f= open(fname, "w")
f.write(str(n)+'\n')

for i in range(n):
    if i != min_pos:
        x = random.uniform(2, 20)
        f.write(str(x)+'\n')
    else:
        f.write(str(1.0)+'\n')
f.close()   