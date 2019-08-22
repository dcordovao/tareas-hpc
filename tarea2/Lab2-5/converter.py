import matplotlib.pyplot as mpimg
import numpy as np
import sys

# Argument 1: 0 if rgb to text and 1 if text to rgb
# Argument 2: name of the file 
# Example calling this program: python converter.py 1 img1
def main():
    
    if len(sys.argv) != 3:
        print("converter.py necesita 2 argumentos!!")
    
    file_type , file_name = sys.argv[1:] 
    file_type = int(file_type)

    if file_type == 0:
        print("Convirtiendo "+file_name+".png a .txt")
        RGBtoTXT(file_name)
    elif file_type == 1:
        print("Convirtiendo "+file_name+".txt a .png")
        TXTtoRGB(file_name)
    else:
        print("Parametro tipo con formato incorrecto!!")

def RGBtoTXT(name):
    img = mpimg.imread(name+'.png')
    M,N,_ = img.shape
    RGB = np.array([img[:,:,i].reshape(M*N) for i in range(3)])
    # img[y,x,c] = RGB[c,x+y*N]
    np.savetxt(name+'.txt', RGB, fmt='%.8f', delimiter=' ', header='%d %d'%(M,N), comments='')

def TXTtoRGB(name):
    RGB = np.loadtxt(name+'.txt', delimiter=' ', skiprows = 1)
    with open(name+'.txt') as imgfile:
        M,N = map(int,imgfile.readline().strip().split())
    img = np.ones((M,N,4))
    for i in range(3):
        img[:,:,i] = RGB[i].reshape((M,N)) 
    mpimg.imsave(name+'_fromfile.png', img)

if __name__ == "__main__":
    main()

# Utilizar nombres sin extension
# Solo se aceptan imagenes en formato png

# Generar archivos de texto:
#RGBtoTXT('img1')

# Generar imagenes
#TXTtoRGB('img_fromfile')