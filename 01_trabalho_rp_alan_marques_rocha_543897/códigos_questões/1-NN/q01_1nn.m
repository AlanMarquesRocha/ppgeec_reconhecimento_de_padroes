% ---------------------------------------------------------------------- %
%            Universidade Federal do Ceará (Campus Sobral)               %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 01 - Reconhecimento de Padrões (BBP1028)           %
%                 Discente: Alan Marques da Rocha (543897)
%     
% ---------------------------------------------------------------------- %


% ---------------------------------------------------------------------- %
%                   Implementação do algoritmo k-NN                      % 
%             Sem a normalização dos dados para k = 1, 2 e 3             %
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

% Definindo o intervalo de k que será implementado (k = 1, 2 e 3):
for k = 1:3
    % Inicializando o vetor das predições:
    predicoes = zeros(size(y));

    % Realizando a validação cruzada com leave-one-out:
    for i = 1:length(y)
        % Separando os dados de treinamento e teste para esta iteração:
        X_treino = X([1:i-1 i+1:end], :);
        y_treino = y([1:i-1 i+1:end]);
        X_teste = X(i, :);

        % Calculando as distâncias Euclidianas entre o exemplo de teste e os 
        % exemplos de treinamento:
        dist_euclidiana = sqrt(sum((X_treino - X_teste).^2, 2));

        % Encontrando as classes dos k vizinhos mais próximos:
        [sorted_distances, indices] = sort(dist_euclidiana);
        k_atual = min(k, length(indices)); % limita k ao número de vizinhos disponíveis
        nn = y_treino(indices(1:k_atual)); % onde nn = nearest neighbors

        % Definindo a classe prevista como a classe mais comum entre os k
        % vizinhos mais próximos:
        predicoes(i) = mode(nn);
    end

    % Calculando a acurácia do método proposto:
    accuracy = sum(predicoes == y)/length(y);
    fprintf('Acurácia do modelo %d-nn com leave-one-out: %.2f%%\n', k, accuracy*100);

    std_deviation = std(predicoes);
    fprintf('Desvio padrão das predições para %d-nn: %.2f\n\n', k, std_deviation);

    % Plotando a matriz de confusão
    C = confusionmat(y, predicoes);
    figure;
    colormap(gray);
    heatmap(C);
    title(sprintf('Matriz de Confusão do modelo %d-nn sem zscore', k));
    xlabel('Classes preditas');
    ylabel('Classes reais');
end






