% ---------------------------------------------------------------------- %
%            Universidade Federal do Ceará (Campus Sobral)               %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 01 - Reconhecimento de Padrões (BBP1028)           %
%                 Discente: Alan Marques da Rocha (543897)
%     
% ---------------------------------------------------------------------- %

%    Implementação do algoritmo Quadratic Discriminant Analysis (QDA)    %
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
x = derma_base(:,1:34);
y = derma_base(:,35);

% Realizando a normalização dos dados através de zscore.
% Calculando a média e desvio padrão de cada coluna dos atributos:

atrib_medias = mean(x);
atrib_desv_padrao = std(x);

% Subtraindo a média de cada coluna dos atributos e dividindo
% pelo desvio padrão:

atrib_norm = (x - atrib_medias) ./ atrib_desv_padrao;
% O operador ./ é usado para realizar a divisão elemento a elemento entre
% dois vetores ou matrizes.

% Atribuindo os atributos normalizados a variável X:
X = atrib_norm;

% Inicializando as variáveis para armazenar as previsões e rótulos 
% verdadeiros:
predicted_labels = zeros(size(y));
true_labels = zeros(size(y));

% Loop de validação cruzada com leave-one-out:
for i = 1:size(X,1)
    % Separando o conjunto de treinamento e teste:
    trainX = X;
    trainX(i,:) = [];
    trainY = y;
    trainY(i) = [];
    testX = X(i,:);
    testY = y(i);

    % Calculando a matriz de covariância de cada classe:
    classes = unique(trainY);
    num_classes = length(classes);
    cov_matrices = cell(num_classes,1);
    for j = 1:num_classes
        cov_matrices{j} = cov(trainX(trainY==classes(j),:));
    end

    % Verificando se alguma matriz de covariância é nula, se for, realiza o
    % ajuste através da variável eps:
    for j = 1:num_classes
        if det(cov_matrices{j}) == 0
            cov_matrices{j} = cov(trainX) + eye(size(trainX,2))*eps;
        end
    end

    % Calculando as probabilidades a priori de cada classe:
    prior_probs = zeros(1,num_classes);
    for j = 1:num_classes
        prior_probs(j) = sum(trainY==classes(j))/length(trainY);
    end

    % Inicializando as variáveis para armazenar as probabilidades de cada 
    % classe:
    class_probs = zeros(size(testY,1),num_classes);

    % Calculando as probabilidades de cada classe para cada amostra de 
    % teste:
    for j = 1:num_classes
        class_probs(:,j) = prior_probs(j) * mvnpdf(testX,mean(trainX(trainY==classes(j),:)),cov_matrices{j});
    end

    % Determinando a classe com a maior probabilidade para cada amostra de
    % teste:
    [~,predicted_labels(i)] = max(class_probs,[],2);
    true_labels(i) = testY;
end

% Calculando a acurácia (acc) do algoritmo QDA:
accuracy = sum(predicted_labels == true_labels)/length(true_labels);
fprintf('A acurácia do modelo QDA foi de: %0.2f%%\n',accuracy*100);

% Criando a matriz de confusão:
confMat = confusionmat(true_labels, predicted_labels);

% Plotando a matriz de confusão:
figure;
confusionchart(confMat);
title('Matriz de Confusão do modelo QDA');
xlabel('Classes preditas');
ylabel('Classes reais');
