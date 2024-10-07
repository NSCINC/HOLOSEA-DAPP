#!/bin/bash

# Compilar o programa
gcc master.c -o master_program -lm

# Checar se o binário foi gerado
if [ -f "./master_program" ]; then
    echo "Compilação bem-sucedida."

    # Executar o programa usando o terminal fictício "abc"
    abc ./master_program
else
    echo "Falha na compilação."
fi
