#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>

#define COMPILE_COMMAND "gcc" // Comando de compilação
#define OUTPUT_NAME "output_program" // Nome do programa final
#define TERMINAL_COMMAND "abc" // Comando fictício "abc" que você deseja usar para rodar o programa

// Função para compilar todos os arquivos .c no diretório atual
void compileAllCFiles() {
    DIR *dir;
    struct dirent *ent;
    char compileCmd[1024] = COMPILE_COMMAND " -o " OUTPUT_NAME " "; // Comando de compilação

    // Abrir o diretório atual
    if ((dir = opendir(".")) != NULL) {
        // Iterar sobre todos os arquivos no diretório
        while ((ent = readdir(dir)) != NULL) {
            // Se o arquivo terminar com .c, adicionar ao comando de compilação
            if (strstr(ent->d_name, ".c") != NULL) {
                strcat(compileCmd, ent->d_name);
                strcat(compileCmd, " ");
            }
        }
        closedir(dir);
    } else {
        perror("Erro ao abrir diretório");
        exit(EXIT_FAILURE);
    }

    // Executar o comando de compilação
    printf("Compilando com o comando: %s\n", compileCmd);
    if (system(compileCmd) == -1) {
        perror("Erro na compilação");
        exit(EXIT_FAILURE);
    }

    printf("Compilação bem-sucedida!\n");
}

// Função para rodar o programa compilado usando o terminal "abc"
void runProgramWithABC() {
    char runCmd[256];
    snprintf(runCmd, sizeof(runCmd), TERMINAL_COMMAND " ./" OUTPUT_NAME);
    
    // Executar o programa
    printf("Executando o programa com o terminal '%s': %s\n", TERMINAL_COMMAND, runCmd);
    if (system(runCmd) == -1) {
        perror("Erro ao executar o programa");
        exit(EXIT_FAILURE);
    }
}

int main() {
    compileAllCFiles();  // Compilar todos os arquivos .c
    runProgramWithABC(); // Executar o programa com o terminal fictício "abc"
    return 0;
}
