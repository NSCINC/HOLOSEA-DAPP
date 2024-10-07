#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Estrutura contendo os argumentos do modelo
typedef struct {
    int dim;
    int nLayers;
    int nHeads;
    int nKvHeads; // Utilize -1 para valores não definidos
    int vocabSize;
    int multipleOf;
    float ffnDimMultiplier;
    float normEps;
    int maxBatchSize;
    int maxSeqLen;
} ModelArgs;

// Implementação da normalização RMS
typedef struct {
    float *weight;
    float eps;
} RMSNorm;

RMSNorm *create_rms_norm(int dim, float eps) {
    RMSNorm *norm = (RMSNorm *)malloc(sizeof(RMSNorm));
    norm->weight = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        norm->weight[i] = 1.0f;
    }
    norm->eps = eps;
    return norm;
}

void destroy_rms_norm(RMSNorm *norm) {
    free(norm->weight);
    free(norm);
}

void rms_norm(RMSNorm *norm, float *x, float *output, int batch_size, int seq_len, int dim) {
    for (int b = 0; b < batch_size; b++) {
        float mean_square = 0;
        for (int i = 0; i < dim; i++) {
            mean_square += x[b * seq_len * dim + i] * x[b * seq_len * dim + i];
        }
        mean_square /= dim;
        float sqrt_mean = sqrt(mean_square + norm->eps);
        for (int i = 0; i < dim; i++) {
            output[b * seq_len * dim + i] = (x[b * seq_len * dim + i] / sqrt_mean) * norm->weight[i];
        }
    }
}

// Implementação da camada de Atenção
typedef struct {
    ModelArgs args;
} Attention;

Attention *create_attention(ModelArgs args) {
    Attention *attn = (Attention *)malloc(sizeof(Attention));
    attn->args = args;
    return attn;
}

void destroy_attention(Attention *attn) {
    free(attn);
}

void attention_forward(Attention *attn, float *x, float *output, int batch_size, int seq_len) {
    // Este é um exemplo simplificado sem implementação real de pesos
    // Aqui você pode adicionar lógica para calcular a atenção
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            // Simulação de cálculo de atenção
            output[b * seq_len + t] = x[b * seq_len + t]; // Passando input como output
        }
    }
}

// Função principal
int main() {
    ModelArgs modelArgs = {4096, 32, 32, -1, -1, 256, 1.0f, 1e-5f, 32, 2048};

    // Criação da camada de atenção
    Attention *attention = create_attention(modelArgs);

    // Exemplo de input de tensor aleatório
    int batch_size = 32;
    int seq_len = 2048;
    int dim = 4096;
    float *x = (float *)malloc(batch_size * seq_len * dim * sizeof(float));
    float *output = (float *)malloc(batch_size * seq_len * sizeof(float));

    // Preenchendo x com valores aleatórios
    for (int i = 0; i < batch_size * seq_len * dim; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 2 - 1; // Valores entre -1 e 1
    }

    // Executando a camada de atenção
    attention_forward(attention, x, output, batch_size, seq_len);

    // Exibir o formato do tensor de saída (apenas exemplo)
    printf("Formato da saída: [%d, %d]\n", batch_size, seq_len);

    // Limpeza
    free(x);
    free(output);
    destroy_attention(attention);
    
    return 0;
}
