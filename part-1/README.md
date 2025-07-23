# Trabalho Final — Parte 1  
## Classificação de Imagens com CNNs (Intel Image Dataset)

- Ernane Ferreira Rocha Junior  
- Quelita Míriam Nunes Ferraz

## 📁 Estrutura do Projeto

```bash
part-1/
│
├── download-dataset.py/          # Script para download automático do Intel Image Dataset 
├── part1_base_model.ipynb/       # Etapas do experimento, modelos e avaliações 
└── README.md/                    # Documentação do projeto
```

## 📊 Dataset: Intel Image Classification
Este projeto utiliza o conjunto de dados Intel Image Classification, originalmente disponibilizado por Puneet Bansal no [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

O dataset contém cerca de 25.000 imagens coloridas com resolução padrão de 150x150 pixels. As imagens representam seis categorias distintas de ambientes naturais, conforme listadas abaixo:

* buildings – imagens de prédios e construções urbanas
* forest – paisagens de floresta
* glacier – formações de gelo e geleiras
* mountain – imagens de montanhas
* sea – paisagens marítimas e costeiras
* street – cenas urbanas de ruas

### 📁 Estrutura dos Dados
O conjunto está organizado em três subconjuntos principais:

```bash
intel-image-classification/
│
├── seg_train/        # Dados de treino (~14.000 imagens)
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
├── seg_test/         # Dados de teste (~3.000 imagens rotuladas)
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
│
└── seg_pred/         # Dados de predição (~7.000 imagens sem rótulo)
```

---

## 🔬 Etapas do Projeto

### 🔹 1. Modelo Base (SimpleCNN)

- Estrutura CNN simples com camadas convolucionais, pooling e totalmente conectadas.
- Usado como baseline, ou seja, o resultado deste modelo é utilizado para comparações futuras para ver se houve melhora ou piora.

### 🔸 2. Modelo Base + Variações de `n_features`

- Experimentos com diferentes larguras de rede alterando o parâmetro `n_feature` (número de filtros).
- Comparação dos efeitos de poucos e muitos filtros nas camadas convolucionais.
- Objetivo desta etapa é investigar como a capacidade do modelo (o número de features que ele pode aprender) afeta a performance na acurácia.

### 🔹 3. Modelo Base + Blocos

- Utiliza a arquitetura base e adiciona blocos de regularização após as camadas de convolução.
- Testar se técnicas de regularização poderiam combater o overfitting observado no modelo base.

### 🔸 4. Nosso Modelo: EfficientStrideCNN

- Modelo personalizado com uso eficiente de **strides** ao invés de pooling para reduzir a dimensão dos mapas de features.
- Otimizado para reduzir a complexidade e aumentar a performance.
- Implementado dentro da classe `Architecture`, que permite modularidade e troca de modelos.
- Avaliado com métricas de precisão, perda e visualização da **matriz de confusão**.

---

## 📊 Avaliação dos Resultados

Todos os modelos foram treinados por 10 épocas. 

### Acurácia de validação e Generalização no conjunto de teste:

Foram treinados e avaliados quatro modelos distintos, variando a arquitetura e o número de filtros. A performance de cada modelo foi medida pela perda (loss) de validação e por um relatório de classificação detalhado ao final de 10 épocas.

| Modelo | Val Loss | Parâmetro | Acurácia (Teste) |
|--------|----------|-----------|------------------|
| Menos filtros 8-16-32	| 0.5781 | 1.334.038 | 81.9% |
| Modelo Base 16-32-64	| 0.5855 | 2.678.694 | 80.8% |
| Modelo com Conv. Kernel 5 | 0.6428 | 5.402.566 | 79.2% |
| Mais filtros 32-64-128 | 0.6531 | 50.726 | 78.4% |
| Base + Blocos (BN/Dropout) | 0.7726 | 2.678.918 | 74.2% |

- O modelo com menos filtros (8-16-32) treinou com eficiência, alcançou menor perda de validação e a maior acurácia no conjunto de testes. Isso indica que ele encontrou um ótimo equilíbrio, aprendendo os padrões necessários sem memorizar o ruído do conjunto de treino (overfitting).

- O aumento no número de filtros piorou a performace, causando um overfitting mais acentuado.

- A adição de blocos de regularização após as camadas de convolução degradou significativamente a performance. Isso sugere que a regularização aplicada foi forte demais, levando o modelo a um estado de underfitting (não conseguiu aprender o suficiente nem mesmo do conjunto de treino).

- O modelo com kernel 5x5 apresentou um resultado intermediário. Embora seja uma técnica válida, para este problema específico não superou a abordagem mais simples e eficiente do modelo com menos filtros.

a menor perda de validação (0.5781), indicando a melhor capacidade de generalização no conjunto de teste entre os modelos avaliados. A acurácia geral deste modelo foi de 81,9%. Os modelos "Base" e "Mais filtros" apresentaram overfitting mais acentuado, como pode ser visto nos gráficos de perda, onde a perda de treinamento continua a diminuir enquanto a de validação estabiliza ou aumenta.

### Análise visual com matriz de confusão:

Para cada um dos cinco modelos, foram gerados um relatório de classificação e uma matriz de confusão. A análise desses resultados permite uma compreensão mais profunda da performance de cada classe.

* Modelo "Menos filtros 8-16-32" (Melhor Performance):

- Acurácia: 81,9%
- Destaques: Apresentou excelente performance para a classe forest (97% de recall) e street (88% de recall). A classe com maior dificuldade de classificação foi glacier, com 74% de recall.

* Modelo "Base + Blocos (BN/Dropout)":

- Acurácia: 74,2%
- Destaques: Este modelo, apesar da regularização com Batch Normalization e Dropout, teve a menor acurácia. Ele se destacou na classe forest (97% de recall) mas teve dificuldades com buildings (63% de recall) e mountain (59% de recall).

A análise visual das matrizes de confusão confirma que a classe glacier é frequentemente confundida com buildings e mountain na maioria dos modelos, indicando uma semelhança visual que dificulta a distinção pela CNN.

### Visualização de ativação de filtros internos com hooks para compreensão interpretável da CNN

Para entender o que a CNN aprende em suas camadas intermediárias, foram utilizados hooks para capturar e visualizar os mapas de ativação das camadas convolucionais.

O processo consiste em:

- Registrar um "gancho" (hook): Uma função é registrada em uma camada específica (ex: conv2). Essa função salva a saída da camada (os mapas de ativação) em um dicionário sempre que a rede processa uma imagem.

- Passar um lote de imagens: Um lote de imagens é processado pelo modelo no modo de avaliação.

- Visualizar as ativações: Os mapas de ativação de uma imagem específica do lote são extraídos. Para cada filtro da camada, o mapa de ativação é exibido como uma imagem em tons de verde, onde áreas mais claras indicam maior ativação.

Essa técnica oferece uma visão interpretável do que cada filtro está detectando. Por exemplo, alguns filtros podem se especializar em detectar bordas, texturas específicas (como folhagens ou rochas) ou formas mais complexas. As ativações do modelo "Meu Modelo com Convolução Kernel 5" foram exploradas, mostrando os diferentes features que a rede aprendeu a identificar.

---

## 🧪 Execução

1. Execute `download-dataset.py` para baixar o dataset.
2. Abra o notebook `part1_base_model.ipynb`.
3. Siga as células sequencialmente para treinar e avaliar os modelos.

---

## 📌 Observações Finais

Este projeto é parte do projeto final da disciplina Aprendizado de Máquina de Mestrado em Engenharia da Computação e Elétrica da UFRN, contendo uso do [notebook](https://github.com/ivanovitchm/PPGEEC2318) disponibilizado pelo professor para a tarefa. Tem como objetivo comparar diferentes abordagens de redes neurais convolucionais aplicadas à tarefa de classificação de imagens naturais. O modelo final reflete uma proposta autoral otimizada com base nos experimentos anteriores.

