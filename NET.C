#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <web3c.h>

#define APOLLO_LOCAL_URL "http://localhost:8545" // URL do seu nó Apollo local

// Função para verificar o número do bloco atual
void check_block_number(web3_client_t *client) {
    char *block_number_str;
    json_t *result = web3_client_eth_block_number(client);

    if (result) {
        block_number_str = json_string_value(result);
        printf("Current block number: %s\n", block_number_str);
        json_decref(result);
    } else {
        printf("Error fetching block number.\n");
    }
}

int main() {
    // Criação do cliente Web3
    web3_client_t *client = web3_client_new(APOLLO_LOCAL_URL);
    
    if (client == NULL) {
        fprintf(stderr, "Error creating Web3 client.\n");
        return 1;
    }

    // Executar operações na blockchain
    check_block_number(client);

    // Liberar recursos
    web3_client_free(client);

    return 0;
}
