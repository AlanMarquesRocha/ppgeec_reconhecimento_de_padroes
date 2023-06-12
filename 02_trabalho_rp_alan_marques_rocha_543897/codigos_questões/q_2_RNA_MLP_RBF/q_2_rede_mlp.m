% ---------------------------------------------------------------------- %
%              Universidade Federal do Ceará (Campus Sobral)             %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 02 - Reconhecimento de Padrões (BBP1028)           %
%                Discente: Alan Marques da Rocha (543897)                %
%                                                                        %
% ---------------------------------------------------------------------- %

%   Implementação de uma Rede Neural Artifical Perceptron Multicamadas   %
%     para encontrar a superfície de decisão da base two_classes.dat     %
%----------------------------------------------------------------------- %

clear
clc

% Carregando a base de dados
data = load('two_classes.dat');
X = data(:, 1:end-1); % Atributos de entrada
y = data(:, end); % Classes


% Parâmetros da rede MLP
hidden_units = 32; % Número de neurônios na camada oculta

% Inicialização dos pesos e bias
input_dim = size(X, 2);
output_dim = 1;
W1 = randn(hidden_units, input_dim);
b1 = randn(hidden_units, 1);
W2 = randn(output_dim, hidden_units);
b2 = randn(output_dim, 1);

% Função de ativação (tangente hiperbólica)
phi = @(x) tanh(x);

% Treinamento da rede
num_epochs = 100;
learning_rate = 0.1; % taxa de aprendizagem

for epoch = 1:num_epochs
    % Forward pass
    Z1 = W1*X' + b1;
    A1 = phi(Z1);
    Z2 = W2*A1 + b2;
    A2 = sign(Z2);
    
    % Cálculo do erro
    error = A2 - y';
    
    % Backward pass
    dZ2 = error;
    dW2 = (1/size(X,1)) * dZ2*A1';
    db2 = (1/size(X,1)) * sum(dZ2, 2);
    dZ1 = (W2'*dZ2) .* (1 - A1.^2);
    dW1 = (1/size(X,1)) * dZ1*X;
    db1 = (1/size(X,1)) * sum(dZ1, 2);
    
    % Atualização dos pesos e bias
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
end

% Predições no conjunto de treinamento
Z1 = W1*X' + b1;
A1 = phi(Z1);
Z2 = W2*A1 + b2;
A2 = sign(Z2);
predictions = A2;

% Calculando a acurácia do modelo
accuracy = sum(predictions' == y) / length(y);
fprintf('Acurácia do modelo: %.2f%%\n', accuracy * 100);

% Plotando a superfície de decisão
figure;
scatter(X(:, 1), X(:, 2), 25, y, 'filled');
hold on;

% Definir os limites do plot
x1_range = linspace(min(X(:, 1)), max(X(:, 1)), 100);
x2_range = linspace(min(X(:, 2)), max(X(:, 2)), 100);
[x1_grid, x2_grid] = meshgrid(x1_range, x2_range);
X_grid = [x1_grid(:), x2_grid(:)];

% Calculando as predições para o grid
Z1_grid = W1*X_grid' + b1;
A1_grid = phi(Z1_grid);
Z2_grid = W2*A1_grid + b2;
A2_grid = sign(Z2_grid);
predictions_grid = A2_grid;

% Plotando a superfície de decisão
contour(x1_grid, x2_grid, reshape(predictions_grid, size(x1_grid)), 'LineWidth', 0.5);
colormap([1 0 0; 0 0 1]); % Vermelho e azul
colorbar;

% Título do gráfico
title('Superfície de Decisão traçada pelo modelo MLP');
xlabel('Atributo 1');
ylabel('Atributo 2');

% Cria a legenda personalizada
h = zeros(3, 1);
h(1) = scatter(NaN, NaN, 'filled', 'MarkerFaceColor', 'red');
h(2) = scatter(NaN, NaN, 'filled', 'MarkerFaceColor', 'blue');
h(3) = plot(NaN, NaN, 'LineWidth', 1.5, 'Color', 'black');
legend(h, {'Classe -1', 'Classe 1', 'Superfície de Decisão'}, 'Location', 'northwest');

set(gca, 'Color', 'k'); % Definir o fundo do gráfico como preto
