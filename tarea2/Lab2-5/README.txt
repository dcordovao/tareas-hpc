Tarea 2 de Computación de Alto Desempeño

Diego Córdova 
Camilo Maldonado

Los archivos contenidos son los siguientes:

tarea2.cpp: El main y la función para hacer los calculos con la CPU
	    -Para compilarlo usar: g++ tarea2.cpp -o tarea1-cpp
	    -El programa recive un argumento que es el archivo de texto a procesar
	    -Ejemplo de llamada: ./tarea-cpp Entrada/imagen.txt
tarea2-2.cpu: El main y la función para hacer los calculos con la GPU de la pregunta 2
	    -Para compilarlo usar: nvcc tarea2-2.cu -o tarea2-cuda
	    -El programa recive un argumento que es el archivo de texto a procesar
	    -Ejemplo de llamada: ./tarea-cuda Entrada/imagen.txt
tarea2-3.cpu: Lo mismo que lo anterior pero para la pregunta 3
converter.py (Auxiliar): Contiene las funciones para transformar de .txt a .png y viceversa
	     - Se puede llamar dandole dos parametros, el primero 0 o 1 dependiendo de si se 
	       quiere convertir cd .png a .txt o viceversa. El segundo parametro es el archivo a convertir
