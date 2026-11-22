# Clasificación semi-supervisada de enfermedades en hojas de tomate — PlantVillage

**Asignatura:** Inteligencia Artificial II  
**Entrega:** Proyecto semestral — Entrega 2 (métodos y resultados preliminares)  
**Integrantes:** Paul Larrea, Roberto Guaña y Victor Suquilanda  
**Docente:** Enrique Vinicio Carrera

## 1. Descripción

Este proyecto compara una línea base de clasificación supervisada con un método de aprendizaje semi-supervisado basado en pseudo-etiquetado y consistencia, empleando imágenes RGB del dataset PlantVillage. Se busca evaluar si las imágenes sin etiquetar contribuyen al desempeño de la clasificación cuando solo se dispone del 25 % de las etiquetas de entrenamiento.

El código está contenido en el notebook [`IA2_Proyecto_Parte2_PlantVillage_Colab.ipynb`](IA2_Proyecto_Parte2_PlantVillage_Colab.ipynb) y está diseñado para ejecutarse en **Google Colab con GPU**.

## 2. Dataset y clases

Fuente: [PlantVillage (mohanty/PlantVillage) en Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage).

Se usa un subconjunto balanceado de **1.000 imágenes RGB** (200 por clase), correspondientes a:

- `Tomato___healthy` — hoja sana.
- `Tomato___Bacterial_spot` — mancha bacteriana.
- `Tomato___Early_blight` — tizón temprano.
- `Tomato___Late_blight` — tizón tardío.
- `Tomato___Tomato_mosaic_virus` — virus del mosaico.

Las imágenes se seleccionan a partir de las particiones oficiales del dataset y se controlan los grupos de hoja mediante `leaf_id` para evitar solapamiento entre entrenamiento, validación y prueba.

| Partición | Imágenes | Por clase |
|---|---:|---:|
| Entrenamiento | 640 | 128 |
| Validación | 160 | 32 |
| Prueba | 200 | 40 |
| **Total** | **1.000** | **200** |

En el escenario de pocas etiquetas se emplean **160 imágenes etiquetadas** y **480 imágenes sin etiquetar**, que proceden únicamente de entrenamiento.

## 3. Experimentos

1. **Supervisado 25 %:** ResNet18 preentrenada con 160 ejemplos etiquetados.
2. **Semi-supervisado 25 %:** la misma arquitectura y 160 ejemplos etiquetados, más 480 imágenes disponibles como no etiquetadas; se emplea una variante simplificada de pseudo-etiquetado y consistencia inspirada en FixMatch.
3. **Supervisado 100 %:** ResNet18 con las 640 imágenes etiquetadas de entrenamiento.

La clasificación se realiza con transferencia de aprendizaje desde ImageNet, entrenando el bloque `layer4` y la capa `fc`. El checkpoint se selecciona mediante F1 macro en validación.

## 4. Entorno y ejecución

**Recomendado:** Google Colab, Python 3 y GPU NVIDIA T4 (u otra GPU disponible).

1. Descarga o abre el notebook `IA2_Proyecto_Parte2_PlantVillage_Colab.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.
3. Ejecuta en orden las celdas del notebook. La primera celda instala las dependencias complementarias.
4. El notebook descarga automáticamente los metadatos y el `data.zip` oficial desde Hugging Face; **no es necesario descargar el dataset manualmente**. La descarga inicial ocupa aproximadamente 2,18 GB y requiere conexión y espacio disponible en Colab.
5. El código extrae solo las 1.000 imágenes seleccionadas, construye las particiones, entrena los tres modelos, evalúa y genera las figuras.
6. En la sección de exportación se crea `resultados_parte2.zip` con los archivos producidos; descárgalo antes de que termine la sesión de Colab.

### Configuración principal

- Semilla: `42`.
- Tamaño de imagen: `224 × 224`.
- Batch size: `32`.
- Épocas máximas: `12`.
- Optimizador: `AdamW`.
- Tasa de aprendizaje de la capa `fc`: `1e-3`.
- Tasa de aprendizaje de `layer4`: `1e-4`.
- Umbral de confianza para pseudo-etiquetas: `0.85`.
- Peso de la pérdida no etiquetada: `0.5`.
- Épocas de warm-up: `3`.
- Early stopping: paciencia de `5` épocas.

Para conocer todas las versiones y parámetros de una ejecución, consulta `resultados/configuracion.json` si se ha incluido en este repositorio.

## 5. Resultados preliminares

Resultados de la ejecución registrada en Google Colab, evaluados sobre las mismas 200 imágenes de prueba:

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Supervisado 25 % | 93,50 % | 93,45 % |
| Semi-supervisado 25 % | 95,50 % | 95,51 % |
| Supervisado 100 % | 98,00 % | 98,01 % |

El semi-supervisado mejora el F1 macro respecto al supervisado 25 % en aproximadamente **2,06 puntos porcentuales**. Se trata de resultados **preliminares**, obtenidos en una única ejecución con semilla fija; no deben interpretarse como garantía de desempeño en cultivos reales.

## 6. Archivos de resultados

Si se incorporan a GitHub, la carpeta `resultados/` puede contener archivos livianos generados por Colab, como:

- `metricas_validacion.csv`, `metricas_test.csv`, `metricas_por_clase.csv`.
- `historial_entrenamiento.csv`, `particiones.csv`, `labeled_25.csv`, `unlabeled_75.csv`.
- `curvas_entrenamiento.png`, `comparacion_f1_test.png`.
- `matrices_confusion_test.png`, `pseudo_etiquetas_aceptadas.png`.
- `eda_distribucion.png`, `eda_ejemplos.png`.
- `configuracion.json`.

**No subir al repositorio:** `data.zip`, las imágenes de PlantVillage, el ZIP completo de resultados, checkpoints `*.pth`, tokens o contraseñas. Los archivos pueden conservarse localmente o en Drive para facilitar su reutilización.

## 7. Limitaciones y trabajo futuro

El dataset presenta fotografías tomadas en condiciones controladas. La Entrega 3 contempla ampliar los experimentos a 10 %, 25 % y 50 % de etiquetas, repetir ejecuciones con varias semillas, comparar umbrales de confianza y profundizar en los errores por clase.

## 8. Referencias

- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science. https://doi.org/10.3389/fpls.2016.01419
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.
- Sohn, K., et al. (2020). *FixMatch: Simplifying semi-supervised learning with consistency and confidence*. NeurIPS.
- PlantVillage: https://huggingface.co/datasets/mohanty/PlantVillage

## 9. Declaración de uso de IA generativa

Durante la preparación del notebook y de la documentación se utilizó asistencia de IA generativa para orientar la programación, estructurar el código y apoyar la redacción. Los experimentos, resultados y métricas reportados corresponden a la ejecución del notebook en Google Colab y deben ser verificados por el equipo.
