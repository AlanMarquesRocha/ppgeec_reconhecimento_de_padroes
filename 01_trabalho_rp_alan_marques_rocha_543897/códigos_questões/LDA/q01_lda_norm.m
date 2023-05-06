% ---------------------------------------------------------------------- %
%            Universidade Federal do Ceará (Campus Sobral)               %
% Programa de Pós-Graduação em Engenharia Elétrica e Computação (PPGEEC) %
% ---------------------------------------------------------------------- %

%            Trabalho 01 - Reconhecimento de Padrões (BBP1028)           %
%                 Discente: Alan Marques da Rocha (543897)
%     
% ---------------------------------------------------------------------- %

%     Implementação do algoritmo Linear Discriminant Analysis (LDA)
%             Utilizando a normalização de dados com zscore              %
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
x = derma_base(:, 1:34);
y = derma_base(:,35);

% Realizando a normalização dos dados através de zscore.
% Calculando a média e desvio padrão de cada coluna dos atributos:

atrib_medias = mean(x);
atrib_desv_padrao = std(x);

% Subtraindo a média de cada coluna dos atributos e dividindo
% pelo desvio padrão.

atrib_norm = (x - atrib_medias) ./ atrib_desv_padrao;
% O operador ./ é usado para realizar a divisão elemento a elemento entre
% dois vetores ou matrizes.

% Atribuindo os atributos normalizados a variável X:
X = atrib_norm;

% Atribuindo a soma das classes a variável n (n = 358):
n = length(y);

% Atribuindo a variável C o número de classes da base de dados (C = 6):
C = max(y);

% Inicializando as varíaveis que receberão os valores da acurácia e erro,
% respectivamente:
acc = 0;
err = 0;

% Inicializando a matriz de confusão (C x C):
conf_mat = zeros(C,C);

% Inicializando os vetores que armazenarão as acurácias e erros de cada iteração:
acc_vec = zeros(n,1);
err_vec = zeros(n,1);

% Loop de validação com leave-one-out:
for i = 1:n
    % Dividindo o conjunto de treinamento e teste:
    X_train = X([1:i-1,i+1:end],:);
    y_train = y([1:i-1,i+1:end],:);
    X_test = X(i,:);
    y_test = y(i,:);
    
    % Calculando as médias e a matriz de covariância para cada classe:
    mu = zeros(C,size(X_train,2));
    Sigma = zeros(size(X_train,2));
    for c = 1:C
        X_c = X_train(y_train == c,:);
        mu(c,:) = mean(X_c);
        Sigma = Sigma + (length(X_c)-1)*cov(X_c);
    end
    Sigma = Sigma/(size(X_train,1)-C);
    
    % Classificando o exemplo de teste:
    y_pred = 0;
    max_prob = -Inf;
    for c = 1:C
        %prob = log(length(X_train(y_train == c,:))/length(y_train)) - 0.5*log(det(Sigma)) - 0.5*(X_test-mu(c,:))*inv(Sigma)*(X_test-mu(c,:))';
        prob = log(length(X_train(y_train == c,:))/length(y_train)) - 0.5*(X_test-mu(c,:))*inv(Sigma)*(X_test-mu(c,:))' - 0.5*log(det(Sigma));

        if prob > max_prob
            max_prob = prob;
            y_pred = c;
        end
    end
    
    % Calculando a acurácia e erro do modelo LDA:
    if y_pred == y_test
        acc = acc + 1;
    else
        err = err + 1;
    end
    
    % Atualizando os valores iniciais da matriz de confusão:
    conf_mat(y_test,y_pred) = conf_mat(y_test,y_pred) + 1;
    
    % Armazenando a acurácia e erro de cada iteração nos vetores acc_vec e err_vec:
    acc_vec(i) = acc/i;
    err_vec(i) = err/i;
end

% Calculando a acurácia e erro médio:
acc_mean = acc/n;
err_mean = err/n;

% Apresentando os resultados obtidos:
disp(['Acurácia média do modelo LDA: ', sprintf('%.2f', acc_mean*100), '%']);

disp(['Erro médio do modelo LDA: ',  sprintf('%.2f', err_mean*100), '%']);

% Plotando a matriz de confusão
figure;
confusionchart(conf_mat);
title('Matriz de Confusão do modelo LDA');
xlabel('Classes preditas');
ylabel('Classes reais');

% Plotando um gráfico de barras das acurácias e erros obtidos em cada iteração:
figure;
bar(1:n, acc_vec);
hold on;
bar(1:n, err_vec);
legend('Acurácia', 'Erro');
title('Acurácia e Erro por iteração do leave-one-out');
xlabel('Iteração');
ylabel('Porcentagem (%)');
