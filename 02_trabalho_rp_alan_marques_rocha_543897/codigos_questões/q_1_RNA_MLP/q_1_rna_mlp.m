% ---------------------------------------------------------------------- %
%              Universidade Federal do Ceará (Campus Sobral)             %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 02 - Reconhecimento de Padrões (BBP1028)           %
%                Discente: Alan Marques da Rocha (543897)                %
%                                                                        %
% ---------------------------------------------------------------------- %

%   Implementação de uma Rede Neural Artifical Perceptron Multicamadas   %
%             com validação através da técnica hold-out                  %
%          para classificação da base de dados Dermatology               %
%----------------------------------------------------------------------- %

% Neste algoritmo, implementa-se uma RNA-MLP do zero, realizando uma busca% 
% dos melhores hiperparâmetros através de testes com números de neurônios %
% e taxa de aprenzidado distintos. A rede possui uma única camada oculta, %
% cuja função de ativação para cada neurônio é a Sigmóide. Entretanto,    %
% utilizou-se a função de ativação softmax nos neurônios de saída, por se %
% tratar de um problema multiclasse.

clear
clc

% Importando a base: dermatology.dat
derma_base = readmatrix('dermatology.dat');

% Pré-processamento dos dados

% Elimina as linhas com elementos desconhecidos (NaN):
derma_base(34,:) = [];
derma_base(34,:) = [];
derma_base(34,:) = [];
derma_base(34,:) = [];

derma_base(259,:) = [];
derma_base(259,:) = [];
derma_base(259,:) = [];
derma_base(259,:) = [];

% Separação dos atributos e das classes dentro de derma_base:

% Atribuindo os atributos padronizados com a técnica z-score a variável X:
X = [(derma_base(:,1:34)-mean(derma_base(:,1:34)))./std(derma_base(:,1:34)), derma_base(:,35)];
y = derma_base(:,35);

% Definindo os parâmetros da validação hold-out com 10 repetições
porcentagem_treino = 0.7;    % Porcentagem dos dados para treinamento
num_repeticoes = 10;      % Número de repetições
acuracias = zeros(num_repeticoes, 1);

% Definindo a grade de hiperparâmetros que serão testados:
camadas_ocultas = [2, 4, 8, 10, 12, 24, 30, 32, 64, 128];    % Possíveis tamanhos da camada oculta
taxa_aprendizado = [0.001, 0.01, 0.1, 0.02, 0.03, 0.0001];  % Possíveis taxas de aprendizado
num_epocas = 250;   % Número de épocas fixo

melhor_acc = 0;  % Recebe a melhor acurácia encontrada
melhor_params = struct();  % Recebe os melhores hiperparâmetros encontrados

for hidden_size = camadas_ocultas
    for learning_rate = taxa_aprendizado
        for repetition = 1:num_repeticoes
            % Dividindo os dados em treinamento e teste
            cv = cvpartition(size(X, 1), 'HoldOut', 1 - porcentagem_treino);
            X_train = X(cv.training, :);
            y_train = y(cv.training, :);
            X_test = X(cv.test, :);
            y_test = y(cv.test, :);

            % Definindo os parâmetros da rede neural MLP
            input_size = size(X_train, 2);
            output_size = max(y_train);

            % Inicialização dos pesos e bias da MLP
            W1 = randn(input_size, hidden_size);
            b1 = zeros(1, hidden_size);
            W2 = randn(hidden_size, output_size);
            b2 = zeros(1, output_size);

            % Variáveis para armazenar a perda em cada época
            loss_history = zeros(num_epocas, 1);

            % Loop de treinamento
            for epoch = 1:num_epocas
                % Forward pass
                z1 = X_train * W1 + b1;
                a1 = sigmoid(z1);
                z2 = a1 * W2 + b2;
                a2 = softmax(z2);

                % Cálculo do erro e gradiente
                loss = cross_entropy_loss(a2, y_train);
                delta2 = a2;
                delta2(sub2ind(size(delta2), 1:size(y_train, 1), y_train')) = delta2(sub2ind(size(delta2), 1:size(y_train, 1), y_train')) - 1;
                
                % Armazena o histórico das perdas
                loss_history(epoch) = loss;

                % Backpropagation
                dW2 = a1' * delta2;
                db2 = sum(delta2);
                delta1 = delta2 * W2';
                delta1 = delta1 .* sigmoid_derivative(a1);
                dW1 = X_train' * delta1;
                db1 = sum(delta1);

                % Atualização dos pesos
                W2 = W2 - learning_rate * dW2;
                b2 = b2 - learning_rate * db2;
                W1 = W1 - learning_rate * dW1;
                b1 = b1 - learning_rate * db1;
            end

            % Teste da rede neural MLP nos dados de teste
            z1 = X_test * W1 + b1;
            a1 = sigmoid(z1);
            z2 = a1 * W2 + b2;
            a2 = softmax(z2);
            [~, y_pred] = max(a2, [], 2);

            % Cálculo da acurácia
            accuracy = sum(y_pred == y_test) / numel(y_test);
            acuracias(repetition) = accuracy;
        end

        % Cálculo da acurácia média
        average_accuracy = mean(acuracias);

        % Verificando se a acurácia atual é melhor do que a melhor encontrada até agora
        if average_accuracy > melhor_acc
            melhor_acc = average_accuracy;
            melhor_params.hidden_size = hidden_size;
            melhor_params.learning_rate = learning_rate;
        end
    end
end

% Imprimindo os melhores hiperparâmetros encontrados
fprintf('Melhores hiperparâmetros:\n');
fprintf('Número de neurônios da camada oculta: %d\n', melhor_params.hidden_size);
fprintf('Taxa de aprendizado: %.3f\n', melhor_params.learning_rate);

% Imprimindo a média da melhor acurácia encontrada dentre os
% hiperparâmetros
fprintf('Acurácia média da rede com os melhores hiperparâmetros: %.2f%%\n', melhor_acc * 100);

% Plota a curva de perda x época
figure;
plot(1:num_epocas, loss_history, 'b-', 'LineWidth', 3);
title('Curva de Perda no treinamento');
xlabel('Número de Épocas');
ylabel('Perda');
grid on;

% Funções de ativação e função de perda:

% Função de ativação sigmoid
function output = sigmoid(x)
    output = 1 ./ (1 + exp(-x));
end

% Calcula a derivada da função de ativação sigmoid
function output = sigmoid_derivative(x)
    output = sigmoid(x) .* (1 - sigmoid(x));
end

% Função de ativação softmax
function output = softmax(x)
    output = exp(x) ./ sum(exp(x), 2);
end

% Cálculo da função de perda de entropia cruzada
function loss = cross_entropy_loss(y_pred, y_true)
    m = size(y_true, 1);
    p = y_pred(sub2ind(size(y_pred), 1:m, y_true'));
    loss = -sum(log(p)) / m;
end
