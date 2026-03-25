# Whisper — Reconocimiento de Voz mediante Supervisión Débil a Gran Escala

### Universidad Autónoma de Occidente
**Facultad de Ingeniería y Ciencias Básicas**
**Maestría en Inteligencia Artificial y Ciencia de Datos**

**Curso:** Procesamiento de Datos Secuenciales con Deep Learning
**Profesor:** Natali Johana Velandia Fajardo

**Estudiantes:**
- Soren Fabricius Acevedo (22500566)
- Ricardo Muñoz Bocanegra (22500246)
- Juan José Bonilla Pinzón (22502052)
- Juan Manuel García Ortiz (22502268)

---

## 1. Resumen (Abstract)

Este proyecto implementa y analiza el modelo **Whisper** de OpenAI, un sistema de reconocimiento automático de voz (ASR) basado en la arquitectura Transformer encoder-decoder. Se evalúan tres variantes del modelo — **tiny** (39M parámetros), **medium** (769M) y **turbo** (809M) — sobre un mismo audio, comparando su precisión de transcripción y rendimiento. Además, se desarrolla una aplicación interactiva en **Streamlit** que visualiza paso a paso el flujo de procesamiento de la arquitectura (desde el espectrograma de Mel hasta la generación autoregresiva de texto), permitiendo observar en tiempo real cómo cada componente del Transformer interviene durante la inferencia. Los resultados muestran las diferencias en calidad de transcripción entre las distintas escalas del modelo y evidencian cómo la arquitectura encoder-decoder procesa señales de audio para producir texto.

---

## 2. Introducción

**Artículo base:** *Robust Speech Recognition via Large-Scale Weak Supervision* (Radford et al., OpenAI, 2022)
- **Paper:** [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **Repositorio original:** [github.com/openai/whisper](https://github.com/openai/whisper)

### Contexto del problema

El reconocimiento automático de voz (ASR) ha sido tradicionalmente abordado con modelos supervisados entrenados en datasets etiquetados manualmente, lo que limita la escala y diversidad de los datos de entrenamiento. Whisper propone un enfoque diferente: entrenar con **680,000 horas de audio** recopiladas de internet con supervisión débil, logrando un modelo robusto que funciona en **99 idiomas** sin necesidad de fine-tuning específico por idioma.

### Motivación

Comprender la arquitectura Transformer aplicada a datos secuenciales de audio, explorando cómo el mecanismo de atención permite al modelo relacionar representaciones de audio con la generación de texto. Además, visualizar de forma interactiva el flujo de procesamiento para facilitar la comprensión de cada componente del modelo.

### Objetivo

Implementar el modelo Whisper en sus variantes tiny, medium y turbo, comparar su desempeño en tareas de transcripción, y desarrollar una aplicación Streamlit que visualice la arquitectura completa del modelo durante el procesamiento de audio en tiempo real.

---

## 3. Marco teórico

### Arquitectura Transformer Encoder-Decoder

Whisper utiliza una arquitectura Transformer sequence-to-sequence compuesta por un **encoder** que procesa el audio y un **decoder** que genera texto de forma autoregresiva.

#### Encoder — Procesamiento del audio

1. **Log-Mel Spectrogram:** El audio se remuestrea a 16 kHz. Se aplica STFT con ventanas de 25 ms (`n_fft=400`) y salto de 10 ms (`hop=160`). Un banco de 80 filtros Mel produce un tensor de forma `(80, 3000)` para 30 segundos de audio.
2. **2× Conv1D + GELU:** Dos capas convolucionales extraen patrones locales del espectrograma, reduciendo los 3000 frames a 1500 y proyectando a `d_model` dimensiones.
3. **Positional Encoding Sinusoidal:** Se suma una señal sinusoidal fija a cada posición para codificar el orden temporal de los frames.
4. **N × [Self-Attention + LayerNorm + FFN]:** Cada frame atiende a todos los demás frames mediante Self-Attention con múltiples cabezas. La salida son los hidden states que representan el audio procesado.

#### Mecanismo de Cross-Attention

El puente entre encoder y decoder se realiza mediante **Cross-Attention**:
- **Q** (Query) proviene del **decoder**: representa "qué necesito saber del audio para generar el siguiente token".
- **K** (Key) y **V** (Value) provienen del **encoder**: representan el audio ya procesado.
- Fórmula: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

Sin Cross-Attention, el decoder generaría texto sin relación con el audio.

#### Decoder — Generación de texto

1. **Tokens de control (Multitasking):** Tokens especiales definen la tarea: `<|startoftranscript|>`, `<|idioma|>`, `<|transcribe|>` o `<|translate|>`. Cambiando estos tokens se obtienen comportamientos distintos sin modificar los pesos.
2. **Learned Positional Encoding:** A diferencia del encoder, el decoder usa embeddings posicionales aprendidos durante el entrenamiento.
3. **Masked Self-Attention:** Una máscara causal impide ver tokens futuros, garantizando que la generación en entrenamiento sea idéntica a la de inferencia.
4. **Cross-Attention (Q ← decoder, K/V ← encoder):** Conecta el audio con la generación de texto.
5. **FFN + Linear + Softmax:** Al final, una capa lineal proyecta al tamaño del vocabulario (51,865 tokens) y softmax convierte en probabilidades.

### Innovaciones clave del paper

- **Escala de entrenamiento:** 680,000 horas de audio con supervisión débil (vs. ~1,000 horas de modelos anteriores).
- **Multitarea con tokens especiales:** Un solo modelo realiza transcripción, traducción y detección de silencio.
- **Zero-shot en 99 idiomas:** Sin fine-tuning por idioma. En inglés alcanza un WER de 2.7%, comparable con sistemas especializados.

---

## 4. Metodología

### Herramientas utilizadas

| Herramienta | Propósito |
|---|---|
| **Python 3** | Lenguaje de programación principal |
| **openai-whisper** | Librería oficial del modelo Whisper |
| **PyTorch** | Backend de deep learning para inferencia |
| **Librosa** | Procesamiento y visualización de audio |
| **Streamlit** | Interfaz web interactiva |
| **Matplotlib** | Visualización de arquitectura y gráficos |
| **pyngrok** | Túnel público para ejecutar Streamlit desde Colab |
| **ffmpeg** | Decodificación de formatos de audio |

### Uso de pesos preentrenados

Se utilizan los pesos preentrenados oficiales de OpenAI descargados automáticamente mediante `whisper.load_model()`. No se realiza fine-tuning; el modelo se usa directamente en modo inferencia. Se evalúan tres variantes:

| Modelo | Parámetros | Capas Encoder | Capas Decoder | d_model | Cabezas |
|---|---|---|---|---|---|
| **tiny** | 39M | 4 | 4 | 384 | 6 |
| **medium** | 769M | 24 | 24 | 1024 | 16 |
| **turbo** | 809M | 32 | 4 | 1280 | 20 |

El modelo **turbo** es una destilación de large-v3 que mantiene el encoder de 32 capas pero reduce el decoder de 32 a 4 capas, logrando velocidad similar a small con calidad cercana a large-v3.

### Proceso de implementación

1. Instalación de dependencias (`openai-whisper`, `ffmpeg`, `librosa`).
2. Carga y preprocesamiento del audio de prueba.
3. Evaluación paso a paso con cada variante del modelo.
4. Desarrollo de la aplicación Streamlit con visualización de arquitectura.
5. Despliegue mediante túnel ngrok para acceso público.

---

## 5. Desarrollo e implementación

### Ejecución del proyecto

**Requisitos previos:**
```bash
pip install openai-whisper librosa soundfile streamlit pyngrok
apt-get install -y ffmpeg
```

**Parte 1 — Demostración del modelo base (Notebook):**

El notebook ejecuta el mismo audio con tres modelos (tiny, medium, turbo), mostrando tanto el flujo paso a paso como la llamada simplificada `transcribe()`.

**Parte 2 — Aplicación Streamlit:**
```bash
streamlit run app.py --server.port 8501 --server.headless true
```

### Carga de pesos preentrenados

Los pesos se descargan automáticamente al llamar `whisper.load_model('nombre_modelo')`. La función verifica si los pesos ya están en caché local (`~/.cache/whisper/`) y los descarga solo si es necesario.

```python
model = whisper.load_model('tiny')  # descarga automática si no está en cache
```

### Proceso de preprocesamiento

1. **Carga del audio:** `whisper.load_audio()` lee el archivo, convierte a `float32` y remuestrea a 16 kHz.
2. **Recorte/padding:** `whisper.pad_or_trim()` ajusta a exactamente 480,000 muestras (30 segundos).
3. **Espectrograma:** `whisper.log_mel_spectrogram()` genera el tensor `(80, 3000)` — 80 bandas de Mel × 3000 frames temporales.

```python
audio = whisper.load_audio('archivo.mp3')
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
```

### Proceso de inferencia

1. **Detección de idioma:** El espectrograma pasa por el encoder y se predice el idioma entre 99 opciones.
2. **Decodificación:** El decoder genera tokens de forma autoregresiva mediante greedy decoding, usando Cross-Attention para consultar los hidden states del encoder en cada paso.

```python
_, probs = model.detect_language(mel)
idioma = max(probs, key=probs.get)

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
```

### Aplicación Streamlit

La aplicación incluye tres pestañas:
- **Transcripción en vivo:** Carga de audio con visualización paso a paso de la arquitectura (el paso activo se ilumina en azul, los completados en verde). Muestra espectrograma, segmentos con timestamps y heatmaps de Cross-Attention.
- **Arquitectura de Whisper:** Explicación detallada de cada componente del encoder y decoder con secciones expandibles.
- **Comparativa de modelos:** Tabla comparativa de parámetros y descripción de innovaciones y limitaciones del paper.

---

## 6. Resultados y análisis

### Comparación entre modelos

Se evaluaron los tres modelos sobre el mismo archivo de audio, observando diferencias en:

- **Precisión de transcripción:** El modelo tiny produce transcripciones con errores frecuentes, mientras que medium y turbo logran resultados significativamente más precisos.
- **Detección de idioma:** Los tres modelos identifican correctamente el idioma del audio, con mayor confianza en modelos más grandes.
- **Tiempo de inferencia:** El modelo tiny es el más rápido. Turbo ofrece un balance óptimo entre velocidad y calidad gracias a su decoder reducido de 4 capas.

### Visualización de la arquitectura

La aplicación Streamlit permite observar en tiempo real:
- **Espectrograma de Mel:** Representación visual del audio como tensor `(80, 3000)`.
- **Flujo de procesamiento:** Diagrama que se ilumina progresivamente mostrando cada etapa del encoder y decoder.
- **Heatmaps de Cross-Attention:** Visualización de las cabezas de atención que revelan qué partes del audio el decoder consulta para generar cada token de texto.

### Métricas de desempeño

| Modelo | Parámetros | Velocidad relativa | Calidad de transcripción |
|---|---|---|---|
| tiny | 39M | Muy rápido | Limitada — errores frecuentes |
| medium | 769M | Moderado | Alta — pocos errores |
| turbo | 809M | Rápido | Alta — cercana a large-v3 |

### Limitaciones observadas

- **Alucinaciones en silencio:** Con ruido puro o silencio, el decoder puede generar texto inexistente (Sección 4.5 del paper).
- **Rendimiento desigual por idioma:** Idiomas con menos datos en internet tienen mayor tasa de error (Figura 3 del paper).
- **Sin memoria entre bloques de 30s:** Cada bloque es independiente, lo que puede causar pérdida de contexto en audios largos (Sección 2.4).
- **Costo computacional:** Los modelos grandes requieren GPU con VRAM significativa (~5 GB+ para medium, ~10 GB para large).

---

## 7. Conclusiones

### Aprendizajes

- La arquitectura Transformer encoder-decoder es altamente efectiva para tareas de sequence-to-sequence como el reconocimiento de voz, permitiendo que el encoder capture representaciones ricas del audio y el decoder las utilice mediante Cross-Attention para generar texto coherente.
- El entrenamiento con supervisión débil a gran escala (680,000 horas) es una estrategia viable para construir modelos robustos sin necesidad de datos etiquetados manualmente, logrando generalización zero-shot en múltiples idiomas.
- La multitarea mediante tokens de control demuestra la flexibilidad de la arquitectura Transformer: un mismo modelo puede transcribir, traducir y detectar idiomas sin modificar sus pesos.

### Limitaciones

- El procesamiento por bloques de 30 segundos impide capturar contexto a largo plazo en audios extensos.
- Los modelos más precisos requieren recursos computacionales significativos (GPU con VRAM dedicada).
- La calidad de transcripción varía considerablemente entre idiomas, reflejando el desbalance en los datos de entrenamiento.

### Posibles mejoras

- Implementar un mecanismo de memoria entre bloques para mantener contexto en audios largos.
- Explorar fine-tuning en datasets específicos del dominio para mejorar la precisión en casos de uso particulares.
- Evaluar técnicas de cuantización para reducir los requisitos de memoria sin sacrificar calidad.
- Incorporar métricas formales de evaluación como WER (Word Error Rate) para una comparación cuantitativa más rigurosa.

---

## 8. Referencias

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023. [Online]. Available: https://arxiv.org/abs/2212.04356

[2] OpenAI, "Whisper," GitHub repository, 2022. [Online]. Available: https://github.com/openai/whisper

[3] A. Vaswani *et al.*, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762

[4] D. S. Park *et al.*, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," in *Proc. Interspeech*, 2019. [Online]. Available: https://arxiv.org/abs/1904.08779

[5] Streamlit, "Streamlit — The fastest way to build data apps," 2023. [Online]. Available: https://streamlit.io

[6] B. McFee *et al.*, "librosa: Audio and Music Signal Analysis in Python," in *Proc. of the 14th Python in Science Conference*, 2015. [Online]. Available: https://librosa.org
