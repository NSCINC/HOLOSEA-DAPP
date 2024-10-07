@const NUM_INPUT = 10         // Número de entradas
@const NUM_HIDDEN = 20        // Número de neurônios na camada oculta
@const NUM_OUTPUT = 1         // Número de saídas
@const MAX_EPOCHS = 1000      // Máximo de épocas de treinamento
@const ERROR_THRESHOLD = 0.01  // Limite de erro para treinamento

// Estrutura para a rede neural
@struct NeuralNetwork {
    weights_input_hidden: double[NUM_INPUT][NUM_HIDDEN]
    weights_hidden_output: double[NUM_HIDDEN][NUM_OUTPUT]
    hidden_layer: double[NUM_HIDDEN]
    output_layer: double[NUM_OUTPUT]
    learning_rate: double
}

// Função de ativação Sigmoid
func sigmoid(x: double) -> double {
    return 1 / (1 + exp(-x))
}

// Derivada da função Sigmoid
func sigmoid_derivative(x: double) -> double {
    return x * (1 - x)
}

// Inicializa a rede neural com pesos aleatórios
func initialize_network(nn: NeuralNetwork) {
    nn.learning_rate = 0.01
    for i in 0 ..< NUM_INPUT {
        for j in 0 ..< NUM_HIDDEN {
            nn.weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1 // Peso aleatório entre -1 e 1
        }
    }
    for j in 0 ..< NUM_HIDDEN {
        for k in 0 ..< NUM_OUTPUT {
            nn.weights_hidden_output[j][k] = ((double)rand() / RAND_MAX) * 2 - 1 // Peso aleatório entre -1 e 1
        }
    }
}

// Função de treinamento da rede neural
func train(nn: NeuralNetwork, input: double[][NUM_INPUT], output: double[][NUM_OUTPUT], num_samples: int) {
    for epoch in 0 ..< MAX_EPOCHS {
        double total_error = 0
        for s in 0 ..< num_samples {
            // Feedforward
            for j in 0 ..< NUM_HIDDEN {
                double activation = 0
                for i in 0 ..< NUM_INPUT {
                    activation += input[s][i] * nn.weights_input_hidden[i][j]
                }
                nn.hidden_layer[j] = sigmoid(activation)
            }

            for k in 0 ..< NUM_OUTPUT {
                double activation = 0
                for j in 0 ..< NUM_HIDDEN {
                    activation += nn.hidden_layer[j] * nn.weights_hidden_output[j][k]
                }
                nn.output_layer[k] = sigmoid(activation)
            }

            // Cálculo do erro
            double error = output[s][0] - nn.output_layer[0]
            total_error += error * error

            // Backpropagation
            double output_delta = error * sigmoid_derivative(nn.output_layer[0])
            for j in 0 ..< NUM_HIDDEN {
                nn.weights_hidden_output[j][0] += nn.learning_rate * output_delta * nn.hidden_layer[j]
            }

            for j in 0 ..< NUM_HIDDEN {
                double hidden_delta = output_delta * nn.weights_hidden_output[j][0] * sigmoid_derivative(nn.hidden_layer[j])
                for i in 0 ..< NUM_INPUT {
                    nn.weights_input_hidden[i][j] += nn.learning_rate * hidden_delta * input[s][i]
                }
            }
        }

        total_error /= num_samples
        if (total_error < ERROR_THRESHOLD) {
            print("Training stopped early at epoch ", epoch)
            break
        }
    }
}

// Função para prever com a rede neural treinada
func predict(nn: NeuralNetwork, input: double[NUM_INPUT]) {
    for j in 0 ..< NUM_HIDDEN {
        double activation = 0
        for i in 0 ..< NUM_INPUT {
            activation += input[i] * nn.weights_input_hidden[i][j]
        }
        nn.hidden_layer[j] = sigmoid(activation)
    }

    for k in 0 ..< NUM_OUTPUT {
        double activation = 0
        for j in 0 ..< NUM_HIDDEN {
            activation += nn.hidden_layer[j] * nn.weights_hidden_output[j][k]
        }
        nn.output_layer[k] = sigmoid(activation)
    }

    print("Prediction: ", nn.output_layer[0])
}

// Função principal
func main() {
    NeuralNetwork nn
    initialize_network(nn)

    // Dados de treinamento fictícios
    double input[2][NUM_INPUT] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
    }
    double output[2][NUM_OUTPUT] = {
        {1},
        {1}
    }

    // Treinar a rede neural
    train(nn, input, output, 2)

    // Realizar previsões
    double new_input[NUM_INPUT] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    predict(nn, new_input)
}

// Chamada da função principal
main()
