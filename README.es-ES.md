

# ParisKV

<p align="center">
  <b>🔥 [ICML'26] ParisKV: Recuperación Rápida y Robusta a la Deriva del Cache KV para LLMs de Contexto Largo</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.07721"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.07721-b31b1b"></a>
  <a href="https://openreview.net/forum?id=wxD4wTYQXt"><img alt="ICML 2026" src="https://img.shields.io/badge/ICML-2026-purple"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.4-green">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c">
  <img alt="Backend" src="https://img.shields.io/badge/backend-PolarANN-black">
</p>

<p align="center">
  <a href="#key-insight">Insight Clave</a> |
  <a href="#why-pariskv">Por qué ParisKV</a> |
  <a href="#quick-start">Inicio Rápido</a> |
  <a href="#evaluation">Evaluación</a> |
  <a href="#repository-contents">Contenido del Repositorio</a>
</p>

ParisKV es un co-diseño algoritmo-sistema para acelerar la decodificación de contexto largo.
En lugar de escanear todo el cache KV en cada paso de decodificación, ParisKV mantiene resúmenes compactos de claves residentes en la GPU, recupera un conjunto de candidatos Top-k de alto recall con PolarANN, reordena los candidatos con estimaciones de producto interno estilo RaBitQ de 4 bits, y busca únicamente los pares KV seleccionados desde la memoria CPU a través de UVA.

<p align="center">
  <img src="assets/pariskv_framework.png" alt="ParisKV framework" width="96%">
</p>

## Key Insight

ParisKV acelera la inferencia de LLMs de contexto largo con recuperación de cache KV robusta a la deriva. A diferencia de los métodos que aprenden centroides solo desde claves de prellenado, las cuales pueden volverse obsoletas durante una generación larga, ParisKV mapea consultas y claves a un espacio de hipersfera unitaria estable y define centroides analíticos y uniformemente distribuidos allí. A medida que evoluciona la decodificación, las claves recién generadas permanecen cerca de al menos un centroide, lo que permite a ParisKV mantener una calidad de recuperación estable ante la deriva de la distribución. Con una canalización de recuperación de grueso a fino nativa para GPU y descarga de cache KV basada en UVA, ParisKV escala a contextos de millones de tokens mientras iguala o incluso supera la atención completa, logrando hasta 2.8x más throughput y una latencia de decodificación 17x / 44x menor que MagicPIG y PQCache.

<p align="center">
  <img src="assets/pariskv_retrieval_drift.png" alt="ParisKV retrieval drift results" width="92%">
  <br>
  <sub>Fig. 1: ParisKV mantiene el Recall@100 estable durante la decodificación larga, mientras que los centroides solo de prellenado se desvían de la distribución de claves en evolución.</sub>
</p>

## Why ParisKV

La decodificación de LLMs de contexto largo está limitada por la memoria: cada token generado requiere leer vectores KV de todos los tokens anteriores. La atención dispersa basada en recuperación resuelve esto seleccionando solo los tokens pasados más relevantes, pero los sistemas existentes de cache KV basados en ANN a menudo sufren de centroides aprendidos obsoletos, sobrecarga de recuperación en el lado CPU o pérdida de precisión por compresión agresiva.

ParisKV está diseñado en torno a tres principios:

- **Recuperación robusta a la deriva.** Normalizar y rotar claves/consultas, y luego usar centroides analíticos independientes de los datos en la esfera unitaria. No se necesita agrupamiento K-means por capa durante el tiempo de prellenado.
- **Búsqueda de grueso a fino nativa para GPU.** Una etapa de votación por colisiones encuentra candidatos gruesos a partir de IDs de libro de códigos compactos; una etapa de reordenamiento fusionado estima las puntuaciones consulta-clave a partir de resúmenes de claves de 4 bits.
- **Descarga KV escalable.** Los tensores KV de precisión completa pueden residir en memoria CPU fijada (pinned), mientras que los kernels de GPU buscan solo los vectores Top-k finales seleccionados a través de Dirección Virtual Unificada.

Según el paper de ParisKV, el sistema:

| Resultado | Resumen |
| --- | --- |
| Calidad | Igual o supera a la atención completa en 7/9 configuraciones de generación larga. |
| Throughput | Hasta 2.8x mayor throughput de decodificación que la atención completa dentro del rango ejecutable de esta última. |
| Latencia a 1M de tokens | Latencia de decodificación 17x menor que MagicPIG y 44x menor que PQCache a escala de 1M de tokens. |
| Escalabilidad | Se ejecuta en contextos largos donde la atención completa se queda sin memoria GPU. |

## Features

- **Libro de códigos PolarANN analítico.** Los centroides direccionales de patrones de signo son fijos, independientes de los datos y de bajo costo para asignar.
- **Preprocesamiento de normalización y rotación SRHT.** Los productos internos se conservan mientras las representaciones se vuelven más estables para la recuperación de subespacios.
- **Recuperación de colisiones multinivel.** La generación de candidatos usa colisiones de subespacios ponderadas para podar la zona de recuperación sin puntuación densa.
- **Metadatos de reordenamiento de 4 bits.** Cada coordenada usa 1 bit de signo más un índice de magnitud de 3 bits; los pesos por subespacio calibran el estimador de producto interno.
- **Kernels CUDA personalizados.** El conteo de colisiones, Top-k por cubos, reordenamiento fusionado, variantes de Top-k adaptativo y la búsqueda KV vía UVA están implementados como kernels de GPU.
- **Soporte para generación larga.** Un diseño de búfer sink/local/actualización mantiene los tokens recientes densos mientras indexa y descarga asíncronamente tokens más antiguos.

## Installation

La configuración probada es CUDA 12.4 con Python 3.10+.

```bash
conda create -n pariskv python=3.10 -y
conda activate pariskv

conda install -y mkl
conda install -c conda-forge libstdcxx-ng -y

pip install -r requirements.txt
pip install "flash-attn>=2.7.0" --no-build-isolation
pip install "flashinfer-python>=0.2.4" -i https://flashinfer.ai/whl/cu124/torch2.5/
```

Las extensiones CUDA se compilan en el primer uso a través del cargador de extensiones de PyTorch.

## Quick Start

La ruta actual del adaptador de modelos es para la familia Qwen a través de `model_hub/qwen.py`.

```python
import torch
from model_hub.qwen import QwenModel

model_name = "Qwen/Qwen3-8B"

llm = QwenModel(
    model_name=model_name,
    max_length=131072,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

if llm.tokenizer.pad_token is None:
    llm.tokenizer.pad_token = llm.tokenizer.eos_token
llm.tokenizer.padding_side = "left"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the key idea behind ParisKV."},
]
prompt = llm.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = llm.tokenizer([prompt], return_tensors="pt", padding=True)
device = llm.layers[0].device

generated_ids, stats = llm.generate(
    attention_type="PolarANN",
    inputs_ids=inputs.input_ids.to(device),
    attention_masks=inputs.attention_mask.to(device),
    max_new_length=512,
    temperature=0.0,
)

print(llm.tokenizer.decode(generated_ids[0], skip_special_tokens=True))
print(stats)
```

Los valores predeterminados de PolarANN específicos del modelo están en `config/*.json`. Puedes pasar un diccionario `attn_config` a `generate(...)` para anular los valores predeterminados en una ejecución.

## How It Works

**1. Prefill: build compact retrieval metadata.**

ParisKV calcula el cache KV, normaliza y rota las claves con SRHT, divide cada clave en subespacios y almacena dos resúmenes residentes en GPU:

- IDs de centroides para recuperación gruesa basada en colisiones
- Códigos direccionales de 4 bits más pesos por subespacio para el reordenamiento

Los vectores KV de precisión completa luego pueden descargarse asíncronamente a la memoria CPU.

<p align="center">
  <img src="assets/pariskv_rotation_codebook.png" alt="ParisKV unit-sphere codebook assignment" width="62%">
  <br>
  <sub>Fig. 3: normalizar-rotar mapea las claves a una esfera unitaria estable donde los centroides analíticos cubren uniformemente el espacio direccional.</sub>
</p>

**2. Decode: retrieve in two stages on GPU.**

Para cada nueva consulta, ParisKV primero activa los centroides direccionales más cercanos en cada subespacio y cuenta las colisiones para producir un grupo de candidatos. Luego reordena esos candidatos con un kernel fusionado de producto interno aproximado de 4 bits y selecciona los índices KV Top-k finales.

<p align="center">
  <img src="assets/pariskv_retrieval_algorithm.png" alt="ParisKV coarse candidate generation and reranking" width="96%">
  <br>
  <sub>Fig. 4: La generación de candidatos nativa para GPU poda por conteos de colisiones, y luego el reordenamiento RSQ-IP de 4 bits selecciona los índices KV Top-k finales.</sub>
</p>

**3. Attention: fetch only selected KV pairs.**

La GPU lee los vectores KV de precisión completa seleccionados desde la memoria CPU fijada vía UVA y calcula la atención sobre tokens sink, tokens locales y tokens recuperados.

## Key Configuration

| Parámetro | Significado | Valor típico |
| --- | --- | --- |
| `sink_size` | Tokens iniciales siempre conservados para atención densa | 4-64 |
| `local_size` | Ventana local de tokens recientes mantenida en la GPU | 256-512 |
| `dynamic_update_interval` | Tokens de decodificación acumulados antes de actualizar los metadatos de recuperación | 256-512 |
| `final_topk` | Número de pares KV seleccionados después del reordenamiento | Depende del benchmark |
| `enable_offload` | Almacenar el KV completo de la zona de recuperación en memoria CPU fijada | `true` / `false` |
| `codebook_path` | Archivo de libro de códigos PolarANN de patrones de signo | `turboquant/codebooks/*.json` |

El cache principal de PolarANN adapta las proporciones de candidatos y colisiones a la longitud de la zona de recuperación:

| Longitud de la zona de recuperación | Proporción de candidatos | Proporción de colisiones |
| --- | --- | --- |
| `< 5K` | 0.50 | 0.50 |
| `5K-10K` | 0.25 | 0.30 |
| `10K-30K` | 0.20 | 0.25 |
| `>= 30K` | 0.10 | 0.20 |

## Evaluation

Ejecuta los comandos desde la raíz del proyecto salvo que se indique lo contrario.

### LongBench-v2

```bash
cd run

# Quick PolarANN smoke run
./run_longbench_v2.sh 1 PolarANN "" easy short cuda:0

# Full PolarANN evaluation
./run_longbench_v2.sh -1 PolarANN
```

### AIME 2025

```bash
export DATA_PATH="/path/to/AIME2025.json"
cd run

# PolarANN pass@8
./run_aime2025.sh PolarANN 8 0.7 0.2 0 "Qwen/Qwen3-8B"
```

Los resultados se escriben en `results/`.

## Codebooks

ParisKV se entrega con libros de códigos pregenerados:

- `turboquant/codebooks/`: libros de códigos direccionales PolarANN de patrones de signo
- `codebooks/`: niveles de magnitud Lloyd-Max para reordenamiento de 4 bits

Para regenerar el libro de códigos de magnitud de 8 niveles predeterminado para `m=8`:

```bash
python run/generate_magnitude_levels.py --levels 8 --m 8 --n_samples 10000000
```

## Repository Contents

- [attn_hub/](attn_hub/) - Envoltorios (wrappers) para PolarANN y FlashAttention.
- [cache_hub/](cache_hub/) - Implementaciones de cache KV y kernels CUDA.
  - [polar_cache.py](cache_hub/polar_cache.py) - Cache principal de ParisKV.
  - [collision/](cache_hub/collision/) y [collision_fused/](cache_hub/collision_fused/) - Kernels de recuperación gruesa basada en colisiones.
  - [rerank/](cache_hub/rerank/) - Kernel de reordenamiento fusionado de 4 bits.
  - [topk/](cache_hub/topk/) - Kernels de Top-k por cubos/radix.
  - [gather/](cache_hub/gather/) - Extensión de recolección (gather) CPU.
  - [gather_trans/](cache_hub/gather_trans/) - Kernel de búsqueda KV H2D vía UVA.
  - [adaptive_k_fused/](cache_hub/adaptive_k_fused/) - Kernel CUDA de Top-k adaptativo.
- [codebooks/](codebooks/) - Niveles de cuantización de magnitud.
- [config/](config/) - Configuraciones PolarANN específicas del modelo.
- [model_hub/](model_hub/) - Integración del modelo Qwen.
- [run/](run/) - Scripts de evaluación para LongBench-v2 y AIME.
- [tests/](tests/) - Pruebas rápidas (smoke) y de rendimiento (throughput).
- [turboquant/codebooks/](turboquant/codebooks/) - Libros de códigos direccionales PolarANN.
- [assets/](assets/) - Figuras del README.

## Notes

- La integración pública de modelos actualmente apunta a la familia Qwen. Otras familias de modelos requieren agregar un adaptador en `model_hub/`.
- Los experimentos estilo paper usan anulaciones específicas del benchmark para el presupuesto de recuperación, ventana local, tokens sink y configuraciones de descarga. Revisa `run/*.sh` y `config/*.json` antes de ejecutar procesos a gran escala.
- Este es código de investigación para experimentos de inferencia de contexto largo; las APIs pueden evolucionar.

## Citation

Si utilizas ParisKV en tu investigación, por favor cita:

```bibtex
@inproceedings{qi2026pariskv,
  title     = {ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs},
  author    = {Qi, Yanlin and Chen, Xinhang and Jiang, Huiqiang and Wang, Qitong and Peng, Botao and Palpanas, Themis},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=wxD4wTYQXt},
  note      = {Poster page: https://icml.cc/virtual/2026/poster/60751}
}
```

## Acknowledgements

ParisKV se construye sobre PyTorch, FlashAttention, FlashInfer, transformadas de Hadamard rápidas y la línea RaBitQ de estimación de producto interno cuantizado.
