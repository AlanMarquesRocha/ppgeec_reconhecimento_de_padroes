% ---------------------------------------------------------------------- %
%            Universidade Federal do Ceará (Campus Sobral)               %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 01 - Reconhecimento de Padrões (BBP1028)           %
%                 Discente: Alan Marques da Rocha (543897)
%     
% ---------------------------------------------------------------------- %

%             Implementação do algoritmo k-NN para k = 1 (1-nn)          %
%----------------------------------------------------------------------- %

clear
clc

% Importando a base: dermatology.dat
derma_base = readmatrix('dermatology.dat');

% Pré-processamento dos dados:

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
X = derma_base(:, 1:34);
y = derma_base(:, 35);

% Realizando a normalização dos dados através de zscore, utilizando a média
% e desvio padrão dos atributos:

atrib_medias = mean(X);
atrib_desv_padrao = std(X);

% Subtraindo a média de cada coluna dos atributos e dividindo
% pelo desvio padrão.

atrib_norm = (X - atrib_medias) ./ atrib_desv_padrao;

% O operador ./ é usado para realizar a divisão elemento a elemento entre
% dois vetores ou matrizes.

% Definindo o número de vizinhos mais próximos a se considerar: 1-nn:
k = 1;

% Inicializando o vetor das predições:
predicoes = zeros(size(y));

% Realizando a validação cruzada com leave-one-out:
for i = 1:length(y)
    % Separando os dados de treinamento e teste para esta iteração:
    X_treino = atrib_norm([1:i-1 i+1:end], :);
    y_treino = y([1:i-1 i+1:end]);
    X_teste = atrib_norm(i, :);
    
    % Calculando as distâncias Euclidianas entre o exemplo de teste e os 
    % exemplos de treinamento:
    dist_euclidiana = sqrt(sum((X_treino - X_teste).^2, 2));
    
    % Encontrando as classes dos k vizinhos mais próximos (k = 1):
    [sorted_distances, indices] = sort(dist_euclidiana);
    k = min(k, length(indices)); % limita k ao número de vizinhos disponíveis
    nn = y_treino(indices(1:k)); % onde nn = nearest neighbors
    
    % Definindo a classe prevista como a classe mais comum entre os k
    % vizinhos mais próximos:
    predicoes(i) = mode(nn);
end

% Calculando a acurácia do método proposto:
accuracy = sum(predicoes == y)/length(y);
fprintf('A acurácia (acc) do modelo 1-nn com leave-one-out: %.2f%%\n', accuracy*100);

std_deviation = std(predicoes);
fprintf('O desvio padrão das predições é: %.2f\n', std_deviation);

% Plotando a matriz de confusão
C = confusionmat(y, predicoes);
figure;
heatmap(C);
title('Matriz de Confusão do modelo 1-nn');
xlabel('Classes preditas');
ylabel('Classes reais');
