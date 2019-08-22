import random

print('Ingrese nombre de archivo a crear:')
fname = input('')
print('Ingrese cantidad de particulas:')
n = int(input(''))
print('Ingrese tamaño de la grilla:')
large = int(input(''))

f= open(fname, "w")
f.write(str(n)+'\n')
for i in range(n):
    x = random.uniform(0, large)
    y = random.uniform(0, large)
    f.write(str(x)+' '+str(y)+'\n')
f.close()