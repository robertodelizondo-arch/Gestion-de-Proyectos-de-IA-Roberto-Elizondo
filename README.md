# Gestion-de-Proyectos-de-IA-Roberto-Elizondo
Actividad de Vision Computacional
# Actividad 4: Implementación y Evaluación de Modelos Preentrenados de Hugging Face

## Descripción del Proyecto
Este proyecto forma parte de la materia de Gestión de Proyectos de Inteligencia Artificial. Su objetivo principal es analizar, seleccionar e implementar modelos preentrenados del ecosistema de **Hugging Face** para un caso de uso específico dentro del ámbito de **Visión por Computadora**, utilizando **Google Colab con GPU** como entorno de prototipado rápido. 

Durante la actividad se evaluaron y compararon tres arquitecturas distintas mediante métricas estándar de desempeño (*Accuracy*, *Precision*, *Recall* y *F1-score*), analizando el equilibrio (*trade-off*) entre precisión y eficiencia computacional (latencia)[cite: 1, 2].

---

## Entorno de Ejecución
* **Plataforma:** Google Colab (Entorno con aceleración por GPU NVIDIA T4)[cite: 1, 2].
* **Lenguaje:** Python 3.x
* **Librerías Principales:** `transformers`, `datasets`, `evaluate`, `torch`, `torchvision`, `scikit-learn`.

---

## Modelos Evaluados
1. **Modelo A (`google/vit-base-patch16-224`):** Vision Transformer enfocado en capturar dependencias globales de la imagen con alta precisión.
2. **Modelo B (`microsoft/resnet-50`):** Arquitectura convolucional clásica optimizada para alta velocidad de inferencia y bajo consumo de recursos.
3. **Modelo C (`facebook/convnext-tiny-224`):** Arquitectura híbrida moderna que equilibra de forma óptima la precisión y la eficiencia.

---

## Estructura del Repositorio
El repositorio se encuentra organizado de la siguiente manera para garantizar la reproducibilidad y el mantenimiento del prototipo[cite: 1, 2]:

```text
├── README.md               # Documentación y descripción general del proyecto
├── requirements.txt        # Dependencias exactas de Python necesarias para la ejecución
├── data/
│   └── dataset_sample.csv  # Muestra del dataset utilizado para las pruebas
├── notebooks/
│   └── Actividad_4.ipynb   # Cuaderno funcional de Google Colab con todo el código de implementación[cite: 1]
├── src/
│   ├── inference.py        # Funciones auxiliares para la carga de modelos y pipelines
│   └── metrics.py          # Scripts de soporte para el cálculo de métricas (Accuracy, Precision, Recall, F1)
└── results/
    └── comparative_table.csv # Resultados exportados de la evaluación de desempeño y latencia
