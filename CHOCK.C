#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_WEIGHTS 3
#define VALID_TOKEN "valid_token"

// Estrutura para armazenar o pedido de investimento
typedef struct {
    char token[50];
    char planName[50];
    int initialInvestment;
    int amount;
    char investorAddress[50];
} InvestmentRequest;

// Estrutura para a rede neural simples
typedef struct {
    double weights[MAX_WEIGHTS];
} SimpleNeuralNetwork;

// Função para inicializar a rede neural
void initializeNeuralNetwork(SimpleNeuralNetwork *nn) {
    srand(time(NULL)); // Inicializa o gerador de números aleatórios
    for (int i = 0; i < MAX_WEIGHTS; i++) {
        nn->weights[i] = (double)rand() / RAND_MAX; // Atribui pesos aleatórios
    }
}

// Função de ativação sigmoide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Função para calcular a saída da rede neural
double feedforward(SimpleNeuralNetwork *nn, double inputData[]) {
    double weightedSum = 0.0;
    for (int i = 0; i < MAX_WEIGHTS; i++) {
        weightedSum += inputData[i] * nn->weights[i];
    }
    return sigmoid(weightedSum);
}

// Função para validar o token NSC
int validateNscToken(const char *token) {
    return strcmp(token, VALID_TOKEN) == 0;
}

// Função para executar o contrato Lua
void runLuaContract(const char *script, const char *function, char *args[]) {
    char command[256];
    snprintf(command, sizeof(command), "lua %s %s", script, function);

    for (int i = 0; args[i] != NULL; i++) {
        strcat(command, " ");
        strcat(command, args[i]);
    }

    printf("Executando: %s\n", command);
    system(command);
}

// Função para adicionar um plano
void addPlan(InvestmentRequest *request) {
    if (!validateNscToken(request->token)) {
        printf("Token NSC inválido\n");
        return;
    }

    char initialInvestmentStr[10];
    snprintf(initialInvestmentStr, sizeof(initialInvestmentStr), "%d", request->initialInvestment);
    char *args[] = {request->planName, initialInvestmentStr, NULL};
    runLuaContract("investment_contract.lua", "add_plan", args);
}

// Função para investir
void invest(InvestmentRequest *request) {
    char amountStr[10];
    snprintf(amountStr, sizeof(amountStr), "%d", request->amount);
    char *args[] = {request->planName, amountStr, request->investorAddress, NULL};
    runLuaContract("investment_contract.lua", "invest", args);
}

int main() {
    // Inicializa a rede neural
    SimpleNeuralNetwork neuralNetwork;
    initializeNeuralNetwork(&neuralNetwork);
    
    double crmData[MAX_WEIGHTS] = {500.0, 100.0, 5.0};
    double result = feedforward(&neuralNetwork, crmData);
    printf("Resultado do processamento de CRM: %.2f\n", result);
    
    // Simulando um pedido de investimento
    InvestmentRequest request;
    strcpy(request.token, "valid_token");
    strcpy(request.planName, "Investment Plan A");
    request.initialInvestment = 1000;
    request.amount = 500;
    strcpy(request.investorAddress, "0x123456789");

    addPlan(&request);
    invest(&request);

    return 0;
}
