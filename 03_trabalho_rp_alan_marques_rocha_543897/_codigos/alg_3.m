% ---------------------------------------------------------------------- %
%              Universidade Federal do Ceará (Campus Sobral)             %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 03 - Reconhecimento de Padrões (BBP1028)           %
%                Discente: Alan Marques da Rocha (543897)                %
%                                                                        %
% ---------------------------------------------------------------------- %

%       Implementação de uma Máquina de Vetor de Suporte (SVM)           %
%     com validação através da técnica hold-out e PCA sem z-score        %
%          para classificação da base de dados Dermatology               %
%----------------------------------------------------------------------- %

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
X = derma_base(:, 1:34);
y = derma_base(:,35);

% Aplicar a técnica PCA para seleção de atributos
n_comp = 10; % Definindo o número desejado de componentes principais
    
comp_pca = pca(X); % Obtendo os componentes principais
X_pca = X * comp_pca(:, 1:n_comp); % Selecionar os componentes principais relevantes

% Calculando os autovalores
auto_valores = var(X_pca, 1);

% Calcular o percentual de informação explicada por cada componente principal
info_comp = (auto_valores / sum(auto_valores)) * 100;

% Apresentando o percentual de informação em cada componente principal
fprintf('----------------------------------------------------\n');
fprintf('Informação distribuída em cada componente principal:\n');
fprintf('----------------------------------------------------\n');

for i = 1:n_comp
    fprintf('Comp. Principal [%d]: %.2f%%\n', i, info_comp(i));
end

% Definir parâmetros da validação hold-out com 10 repetições
repeticoes = 10;
percent_treino = 0.7;
percent_teste = 0.3;

acuracias = zeros(repeticoes, 1);

for i = 1:repeticoes
    % Dividir os dados em conjuntos de treinamento e teste
    cv = cvpartition(size(X_pca, 1), 'HoldOut', percent_teste);
    X_treino = X_pca(training(cv), :);
    y_treino = y(training(cv), :);
    X_teste = X_pca(test(cv), :);
    y_teste = y(test(cv), :);
    
    % Treinar a SVM multiclasse usando a técnica One-vs-All
    svm = fitcecoc(X_treino, y_treino);
    
    % Fazer previsões nos dados de teste
    y_pred = predict(svm, X_teste);
    
    % Calcular a acurácia em cada iteração
    acuracia = sum(y_pred == y_teste) / numel(y_teste);
    acuracias(i) = acuracia;
    
    % Imprimir a acurácia da repetição atual
    
    fprintf('\n Acurácia da repetição [%d]: %.2f%%\n', i, acuracia * 100);
end

% Calcular a acurácia média
acuracia_media = mean(acuracias);

fprintf('------------------------------------\n');

% Imprimir a acurácia média
fprintf('Acurácia média: %.2f%%\n', acuracia_media * 100);

fprintf('------------------------------------\n');
