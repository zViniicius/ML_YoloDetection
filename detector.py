## 1. Configuração e Importações

# Instale os pacotes necessários
# pip install torch torchvision opencv-python matplotlib PyYAML

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

## 2. Definição da Classe ObjectDetector

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov5s', confidence: float = 0.5):
        """
        Inicializa o detector de objetos com modelo e limite de confiança especificados.
        
        Argumentos:
            model_name (str): Nome do modelo YOLOv5 a ser usado
            confidence (float): Limite de confiança para detecções
        """
        self.logger = self._setup_logger()
        self.confidence = confidence
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'Using device: {self.device}')
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name)
            self.model.conf = confidence
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f'Error loading model: {e}')
            raise

    def _setup_logger(self) -> logging.Logger:
        """
        Configura as configurações de registro do sistema.
        """
        logger = logging.getLogger('ObjectDetector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def load_image(self, image_path: Union[str, Path]) -> Tuple[bool, Union[None, np.ndarray]]:
        """
        Carrega uma imagem do caminho especificado.
        
        Argumentos:
            image_path: Caminho para o arquivo de imagem
            
        Retorna:
            Tupla contendo status de sucesso e array da imagem se bem-sucedido
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                self.logger.error(f'Failed to load image: {image_path}')
                return False, None
            return True, img
        except Exception as e:
            self.logger.error(f'Error loading image: {e}')
            return False, None

    def detect_objects(self, image: np.ndarray) -> torch.Tensor:
        """
        Realiza detecção de objetos na imagem de entrada.
        
        Argumentos:
            image: Array da imagem de entrada
            
        Retorna:
            Resultados da detecção
        """
        try:
            results = self.model(image)
            return results
        except Exception as e:
            self.logger.error(f'Error during detection: {e}')
            raise

    def save_detections(self, results: torch.Tensor, output_path: Union[str, Path]) -> None:
        """
        Salva os resultados da detecção com caixas delimitadoras.
        
        Argumentos:
            results: Resultados da detecção do modelo
            output_path: Caminho para salvar a imagem de saída
        """
        try:
            output_img = results.render()[0]
            cv2.imwrite(str(output_path), output_img)
            self.logger.info(f'Saved detection results to: {output_path}')
        except Exception as e:
            self.logger.error(f'Error saving detection results: {e}')
            raise

## 3. Configuração

# Configurar parâmetros
config = {
    'input_path': 'image.jpg',  # Substitua pelo caminho da sua imagem
    'output_dir': 'detection_out',
    'model_name': 'yolov5s',
    'confidence': 0.1
}

# Criar diretório de saída
output_path = Path(config['output_dir'])
output_path.mkdir(parents=True, exist_ok=True)

## 4. Inicializar Detector

# Criar instância do detector
detector = ObjectDetector(model_name=config['model_name'], 
                        confidence=config['confidence'])

## 5. Carregar e Processar Imagem

# Carregar imagem
success, image = detector.load_image(config['input_path'])
if not success:
    print("Failed to load image")
else:
    # Realizar detecção
    results = detector.detect_objects(image)
    
    # Gerar caminhos de saída com timestamp
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    detection_path = output_path / f'detections_{timestamp}_{config["input_path"]}'
    
    # Salvar resultados
    detector.save_detections(results, detection_path)

## 6. Análise dos Resultados

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
