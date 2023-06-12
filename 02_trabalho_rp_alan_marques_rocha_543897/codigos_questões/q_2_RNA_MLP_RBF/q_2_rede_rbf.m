% ---------------------------------------------------------------------- %
%              Universidade Federal do Ceará (Campus Sobral)             %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 02 - Reconhecimento de Padrões (BBP1028)           %
%                Discente: Alan Marques da Rocha (543897)                %
%                                                                        %
% ---------------------------------------------------------------------- %

%       Implementação de uma Rede de Função de Base Radial (RBF)         %
%     para encontrar a superfície de decisão da base two_classes.dat     %
%----------------------------------------------------------------------- %

clear
clc

% Carregando a base de dados
data = load('two_classes.dat');
x = data(:, 1:end-1); % Atributos de entrada
y = data(:, end); % Classes

% Padronização dos dados com a técnica z-score:
X = zscore(x);

% Parâmetros da rede RBF
num_neurons = 16; % Número de neurônios da camada oculta

% Encontrar os centróides usando k-means
[idx, centroids] = kmeans(X, num_neurons);

% Calculando as distâncias entre os exemplos e os centróides (Distância
% Euclidiana)
D = pdist2(X, centroids);

% Definindo a função de ativação (tangente Hiperbólica)
phi = @(x) tanh(x);

% Calcula a matriz de ativação
A = phi(D);

% Adicionar o termo de polarização
A = [ones(size(A, 1), 1) A];

% Treinamento da rede usando regressão logística
w = pinv(A)*y;

% Calculando as predições do modelo
predictions = sign(A*w);

% Calculando a acurácia da rede no conjunto de treinamento.
accuracy = sum(predictions == y) / length(y);
fprintf('Acurácia do modelo: %.2f%%\n', accuracy * 100);

% Plotando a superfície de decisão
figure;
scatter(X(:, 1), X(:, 2), 25, y, "filled");
hold on;

% Definir os limites do plot
x1_range = linspace(min(X(:, 1)), max(X(:, 1)), 100);
x2_range = linspace(min(X(:, 2)), max(X(:, 2)), 100);
[x1_grid, x2_grid] = meshgrid(x1_range, x2_range);
X_grid = [x1_grid(:), x2_grid(:)];

% Calcular a matriz de distâncias para o grid
D_grid = pdist2(X_grid, centroids);
A_grid = phi(D_grid);
A_grid = [ones(size(A_grid, 1), 1) A_grid];

% Calcular as predições para o grid
predictions_grid = sign(A_grid*w);

% Plotando a superfície de decisão
contour(x1_grid, x2_grid, reshape(predictions_grid, size(x1_grid)), 'LineWidth', 0.5);
colormap([1 0 0; 0 0 1]); % Vermelho e azul
colorbar;

% Título do gráfico
title('Superfície de Decisão traçado pelo modelo RBF');
xlabel('Atributo 1');
ylabel('Atributo 2');

% Cria a legenda personalizada
h = zeros(3, 1);
h(1) = scatter(NaN, NaN, 'filled', 'MarkerFaceColor', 'red');
h(2) = scatter(NaN, NaN, 'filled', 'MarkerFaceColor', 'blue');
h(3) = plot(NaN, NaN, 'LineWidth', 1.5, 'Color', 'black');
legend(h, {'Classe -1', 'Classe 1', 'Superfície de Decisão'}, 'Location', 'northwest');

set(gca, 'Color', 'k'); % Definir o fundo do gráfico como preto
