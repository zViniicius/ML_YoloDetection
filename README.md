# Detector de Objetos YOLOv5 com PyTorch

Este projeto utiliza o modelo YOLOv5 para detecção de objetos em imagens. Ele é implementado em Python com a biblioteca PyTorch e pode ser executado facilmente em um ambiente como o Google Colab.

## Funcionalidades

- Realizar a detecção de objetos usando o modelo YOLOv5.
- Exibir e salvar os resultados com caixas delimitadoras ao redor dos objetos detectados.
- Registrar as classes de objetos detectados e suas contagens.

## Como Usar no Google Colab

### 1. Preparar o Ambiente

Primeiramente, abra o Google Colab e crie um novo notebook.

Em seguida, instale as dependências necessárias executando a célula abaixo:

```python
# Instale as bibliotecas necessárias
!pip install torch torchvision opencv-python matplotlib PyYAML
```

### 2. Carregar o Código

Copie e cole o código abaixo em uma célula do seu notebook para importar as bibliotecas e definir a classe `ObjectDetector`.

```python
# Importar bibliotecas necessárias
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple
import logging
import yaml
from datetime import datetime
import numpy as np

# Definir a classe ObjectDetector aqui (como descrito no código acima)
```

### 3. Configuração de Parâmetros

Configure os parâmetros de entrada, como o caminho para a imagem de entrada e o diretório de saída para os resultados da detecção. Você pode configurar os valores diretamente no código ou carregar imagens do seu Google Drive ou de uma URL.

```python
# Configurar parâmetros
config = {
    'input_path': 'image.jpg',  # Substitua pelo caminho da sua imagem
    'output_dir': 'detection_out',
    'model_name': 'yolov5s',
    'confidence': 0.5
}
```

Caso queira carregar imagens do seu Google Drive, basta usar o seguinte código:

```python
from google.colab import drive
drive.mount('/content/drive')

# Caminho para a imagem no seu Google Drive
config['input_path'] = '/content/drive/MyDrive/images/image.jpg'
```

### 4. Inicializar o Detector

Agora, crie uma instância do detector de objetos, passando o nome do modelo (por exemplo, `yolov5s` para a versão pequena do modelo YOLOv5) e o valor de confiança desejado (padrão é 0.5, mas você pode ajustar).

```python
# Criar instância do detector
detector = ObjectDetector(model_name=config['model_name'], 
                        confidence=config['confidence'])
```

### 5. Carregar e Processar Imagem

Em seguida, carregue a imagem de entrada e execute a detecção de objetos.

```python
# Carregar imagem
success, image = detector.load_image(config['input_path'])
if not success:
    print("Falha ao carregar a imagem.")
else:
    # Realizar detecção
    results = detector.detect_objects(image)
    
    # Gerar caminho de saída com timestamp
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    detection_path = Path(config['output_dir']) / f'detections_{timestamp}_{Path(config["input_path"]).name}'
    
    # Salvar resultados
    detector.save_detections(results, detection_path)
```

### 6. Analisar os Resultados

Após a detecção, você pode analisar as classes de objetos detectados e suas contagens. O código exibe as classes detectadas e quantas vezes cada uma aparece.

```python
# Exibir resumo da detecção
if 'results' in locals():
    print("\nResumo da detecção:")
    print(f"Total de objetos detectados: {len(results.pred[0])}")
    
    # Exibir classes detectadas e suas contagens
    classes = results.pred[0][:, -1].cpu().numpy()
    unique_classes, counts = np.unique(classes, return_counts=True)
    
    print("\nClasses detectadas:")
    for class_idx, count in zip(unique_classes, counts):
        class_name = results.names[int(class_idx)]
        print(f"{class_name}: {count}")
```

## Como Funciona

1. **Inicialização do Modelo**: O modelo YOLOv5 é carregado usando o PyTorch a partir do repositório oficial da Ultralytics. O modelo é carregado para o dispositivo (GPU ou CPU), e a confiança mínima para as detecções é configurada.
   
2. **Carregamento da Imagem**: A função `load_image` carrega a imagem a partir de um caminho especificado, retornando a imagem como um array NumPy.

3. **Detecção de Objetos**: A função `detect_objects` executa a detecção utilizando o modelo YOLOv5, que retorna os resultados em forma de tensores.

4. **Renderização dos Resultados**: Após a detecção, a função `save_detections` desenha as caixas delimitadoras e rótulos dos objetos na imagem e a salva no diretório de saída.

5. **Exibição de Resultados**: O código também permite visualizar a imagem com as caixas delimitadoras e exibir um resumo das classes detectadas e suas quantidades.

## Estrutura do Projeto

```
Detector YOLOv5
│
├── detector.py          # Código principal com a classe ObjectDetector
├── detection_out/       # Diretório de saída para imagens com detecção
│   └── detections_01-01-2024_14-30-45_image.jpg  # Imagem com objetos detectados
├── image.jpg            # Imagem de entrada para detecção
└── README.md            # Este arquivo
```

## Considerações Finais

Este projeto pode ser facilmente adaptado para diferentes tipos de modelos e cenários de detecção. Você pode mudar o modelo YOLOv5 para uma versão maior (como `yolov5m` ou `yolov5l`) para obter resultados mais precisos ou alterar os parâmetros de entrada e saída conforme necessário.
