#include <stdio.h>
#include <string.h>

#define MAX_INVESTORS 100 // Definindo um limite para o número de investidores

typedef struct {
    char *investors[MAX_INVESTORS];
    int authorizedInvestors[MAX_INVESTORS]; // 1 para autorizado, 0 para não autorizado
    long balances[MAX_INVESTORS]; // Saldo de cada investidor
    long investedAmount[MAX_INVESTORS]; // Montante investido por cada investidor
    int investorCount; // Contagem de investidores
} InvestmentContract;

void invest(InvestmentContract *contract, const char *investor, long amount) {
    if (amount <= 0) {
        printf("Investment amount must be greater than zero\n");
        return;
    }

    int investorIndex = -1;

    // Verifica se o investidor está na lista de investidores
    for (int i = 0; i < contract->investorCount; i++) {
        if (strcmp(contract->investors[i], investor) == 0) {
            investorIndex = i;
            break;
        }
    }

    if (investorIndex == -1) {
        printf("Investor '%s' not found in the contract\n", investor);
        return;
    }

    // Verifica se o investidor está autorizado
    if (!contract->authorizedInvestors[investorIndex]) {
        printf("Investor '%s' is not authorized to invest\n", investor);
        return;
    }

    // Verifica se o saldo é suficiente
    if (amount > contract->balances[investorIndex]) {
        printf("Insufficient balance for investor '%s'. Current balance: %ld\n", investor, contract->balances[investorIndex]);
        return;
    }

    // Atualiza os saldos e montantes investidos
    contract->balances[investorIndex] -= amount;
    contract->investedAmount[investorIndex] += amount;
    printf("Investment of %ld made by '%s'\n", amount, investor);
}

// Função para exemplo de uso
void initializeContract(InvestmentContract *contract) {
    // Inicialização do contrato e exemplo de investidores
    contract->investorCount = 3;

    contract->investors[0] = "Alice";
    contract->authorizedInvestors[0] = 1; // Autorizado
    contract->balances[0] = 1000; // Saldo inicial

    contract->investors[1] = "Bob";
    contract->authorizedInvestors[1] = 0; // Não autorizado
    contract->balances[1] = 500; // Saldo inicial

    contract->investors[2] = "Charlie";
    contract->authorizedInvestors[2] = 1; // Autorizado
    contract->balances[2] = 1500; // Saldo inicial
}

int main() {
    InvestmentContract contract;
    initializeContract(&contract);

    // Testando a função invest
    invest(&contract, "Alice", 200);   // Investimento válido
    invest(&contract, "Bob", 100);     // Investidor não autorizado
    invest(&contract, "Charlie", 2000); // Saldo insuficiente
    invest(&contract, "Dave", 100);    // Investidor não encontrado

    return 0;
}
