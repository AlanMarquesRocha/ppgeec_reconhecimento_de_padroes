% ---------------------------------------------------------------------- %
%            Universidade Federal do Ceará (Campus Sobral)               %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 01 - Reconhecimento de Padrões (BBP1028)           %
%                 Discente: Alan Marques da Rocha (543897)
%     
% ---------------------------------------------------------------------- %


clear
clc

% Importando a base: dermatology.dat
derma_base = readmatrix('dermatology.dat');

% Pré-processamento dos dados:

% elimina as linhas com elementos desconhecidos
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

% Realizando a normalização dos dados através de zscore.
% Calculando a média e desvio padrão de cada coluna dos atributos:

atrib_medias = mean(X);
atrib_desv_padrao = std(X);

% Subtraindo a média de cada coluna dos atributos e dividindo
% pelo desvio padrão.

atrib_norm = (X - atrib_medias) ./ atrib_desv_padrao;

% O operador ./ é usado para realizar a divisão elemento a elemento entre
% dois vetores ou matrizes.

% Definindo o número de vizinhos mais próximos a considerar: 1-nn
k = 1;

% Inicializando o vetor das predições:
predicted = zeros(size(y));

% Realizando a validação cruzada com leave-one-out:
for i = 1:length(y)
    % Separando os dados de treinamento e teste para esta iteração:
    X_train = atrib_norm([1:i-1 i+1:end], :);
    y_train = y([1:i-1 i+1:end]);
    X_test = atrib_norm(i, :);
    
    % Calculando as distâncias entre o exemplo de teste e os exemplos de treinamento
    dist_euclidiana = sqrt(sum((X_train - X_test).^2, 2));
    
    % Encontrando as classes dos k vizinhos mais próximos
    [sorted_distances, indices] = sort(dist_euclidiana);
    k = min(k, length(indices)); % limita k ao número de vizinhos disponíveis
    nn = y_train(indices(1:k)); % onde nn = nearest neighbors
    
    % Definindo a classe prevista como a classe mais comum entre os k
    % vizinhos mais próximos:
    predicted(i) = mode(nn);
end

% Calculando a acurácia do método proposto:
accuracy = sum(predicted == y)/length(y);
fprintf('Acurácia do modelo 1-nn com leave-one-out: %.2f%%\n', accuracy*100);

% Plotando a matriz de confusão
C = confusionmat(y, predicted);
figure;
heatmap(C);
title('Matriz de Confusão do modelo 1-nn');
xlabel('Classe prevista');
ylabel('Classe real');