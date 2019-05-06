# Programa que al ejecutarse compila y ejecuta los archivos del programa

import subprocess
print("Compilando tarea1.cpp")
subprocess.call(["g++", "tarea1.cpp","-o", "tarea1-cpp"])
print("Compilando tarea1.cu")
subprocess.call(["nvcc", "tarea1.cu","-o", "tarea1-cuda"])

input_files = ["Entrada/images1.txt",
                "Entrada/images2.txt",
                "Entrada/images3.txt",
                "Entrada/images4.txt",
                "Entrada/images5.txt",
                "Entrada/images6.txt"]

for in_file in input_files:
    print("Executando tarea1-cpp con input ",in_file)
    subprocess.call(["./tarea1-cpp", in_file])

for in_file in input_files:
    print("Executando tarea1-cuda con input ",in_file)
    subprocess.call(["./tarea1-cuda", in_file])