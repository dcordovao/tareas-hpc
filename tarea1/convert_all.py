# Programa que solo convierte todos los resultados a .png

import subprocess

input_files = ["images1",
                "images2",
                "images3",
                "images4",
                "images5",
                "images6"]

for in_file in input_files:
    subprocess.call(["python","converter.py","1", "result_"+in_file])
    subprocess.call(["python","converter.py","1", "result_cuda_"+in_file])