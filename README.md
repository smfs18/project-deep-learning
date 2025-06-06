# Classificação de Imagens com Transfer Learning usando VGG16

Neste projeto, eu demonstro uma técnica poderosa e comum em visão computacional: **transfer learning**. Eu utilizei o modelo **VGG16**, pré-treinado no extenso dataset ImageNet, e o adaptei para uma nova tarefa específica: classificar imagens em uma de seis categorias, alcançando até **96% de acurácia** no conjunto de validação.

Eu dividi estrategicamente o processo de treinamento em duas fases principais para maximizar o desempenho e a estabilidade:

1.  **Extração de Características:** Comecei congelando a base convolucional do modelo VGG16. Isso me permitiu usar suas características já aprendidas sem alterá-las. Em seguida, treinei apenas o novo classificador personalizado que adicionei no topo. Isso adaptou rapidamente o modelo para as classes do meu novo dataset.

2.  **Ajuste Fino (Fine-Tuning):** Após o treinamento inicial, eu descongelei o modelo inteiro e continuei o treinamento com uma taxa de aprendizado muito baixa. Este passo ajustou ligeiramente os pesos pré-treinados para se adequarem melhor às nuances do meu dataset específico, melhorando ainda mais a acurácia.

## ⚙️ Tecnologias que Utilizei

* **TensorFlow & Keras:** Para construir e treinar o modelo de deep learning.
* **VGG16:** A rede neural convolucional (CNN) pré-treinada que usei como modelo base.
* **ImageDataGenerator:** Para um carregamento de dados eficiente e aumento de dados em tempo real.

## 🧠 Arquitetura do Modelo

Eu construí o modelo final empilhando novas camadas sobre a base pré-treinada VGG16.

1.  **Modelo Base (VGG16):** Carreguei o modelo VGG16, descartando sua camada de classificação original (`include_top=False`), mas mantendo os pesos aprendidos com o ImageNet.

2.  **Congelamento de Camadas:** Inicialmente, todas as camadas na base VGG16 foram congeladas (`base_model.trainable = False`) para que seus pesos não fossem atualizados durante a primeira fase do treinamento.

3.  **Classificador Personalizado:** Adicionei minha própria cabeça de classificação:
    * Uma camada `GlobalAveragePooling2D` para reduzir as dimensões espaciais dos mapas de características para um único vetor, reduzindo drasticamente o número de parâmetros.
    * Uma camada final `Dense` com 6 unidades (uma para cada classe de destino) e uma função de ativação `softmax` para obter a distribuição de probabilidade entre as classes.

## 📦 Preparação e Aumento de Dados

Para tornar o modelo mais robusto e evitar overfitting, apliquei aumento de dados (data augmentation) nas imagens de treinamento em tempo real usando o `ImageDataGenerator`. As transformações aplicadas incluem:

* Rotações aleatórias
* Zoom aleatório
* Deslocamentos horizontais e verticais aleatórios
* Inversões horizontais aleatórias

Os dados de validação não foram aumentados; eu apenas reescalei, assim como fiz com os dados de treinamento.

## 🚀 Processo de Treinamento

### Fase 1: Extração de Características

Primeiro, compilei o modelo com o otimizador `adam` e a perda `CategoricalCrossentropy`. Eu o treinei por 20 épocas. Nesta fase, apenas os pesos das camadas `GlobalAveragePooling2D` e `Dense` foram atualizados. Esta fase de treinamento inicial me permitiu alcançar uma acurácia sólida de **94%** no conjunto de validação.

```python
# Congelar o modelo base
base_model.trainable = False

# Compilar o modelo
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

# Treinar apenas as camadas superiores
model.fit(train_it,
          validation_data=valid_it,
          epochs=20)
```

### Fase 2: Ajuste Fino (Fine-Tuning)

Em seguida, descongelei o modelo base para permitir que todos os seus pesos fossem treináveis. Recompilei o modelo com uma taxa de aprendizado muito baixa (`0.0001`) e o otimizador `RMSprop`. Usei uma taxa de aprendizado baixa, o que é crucial para não destruir as características valiosas aprendidas com o ImageNet. Treinei o modelo por mais 20 épocas. Este passo elevou a acurácia de validação para impressionantes **96%**.

```python
# Descongelar o modelo base para permitir o ajuste fino
base_model.trainable = True

# Recompilar com uma taxa de aprendizado muito baixa
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = 0.0001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics =[keras.metrics.CategoricalAccuracy()])

# Continuar o treinamento do modelo inteiro
model.fit(train_it,
          validation_data=valid_it,
          epochs=20)
```

## 📈 Resultados

A minha abordagem de treinamento em duas fases produziu excelentes resultados:

* **Acurácia de Extração de Características (Fase 1):** **94%** no conjunto de validação.
* **Acurácia de Ajuste Fino (Fase 2):** **96%** no conjunto de validação.

Isso demonstra o poder do transfer learning e como uma segunda fase de ajuste fino pode proporcionar um aumento adicional de desempenho, adaptando as características aprendidas mais de perto ao dataset específico.

## 📋 Como Usar

1.  **Clonar o Repositório:**
    ```bash
    git clone <url-do-seu-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Organizar os Dados:** Certifique-se de que seu conjunto de dados de imagem esteja estruturado da seguinte maneira:
    ```
    dataset/
    ├── train/
    │   ├── classe_1/
    │   └── classe_2/
    └── valid/
        ├── classe_1/
        └── classe_2/
    ```

3.  **Atualizar os Caminhos dos Arquivos:** No script Python, altere os caminhos nas chamadas `flow_from_directory` para que apontem para seus diretórios `train` e `valid`.
    ```python
    train_it = datagen_train.flow_from_directory(
        'caminho/para/seu/dataset/train',
        # ... outros parâmetros
    )
    
    valid_it = datagen_valid.flow_from_directory(
        'caminho/para/seu/dataset/valid',
        # ... outros parâmetros
    )
    ```

4.  **Executar** o script para iniciar o processo de treinamento.
