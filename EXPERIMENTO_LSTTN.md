# Propuesta experimental: LSTTN de una semana

## Pregunta

¿Qué dimensión de representación produce mejores características para una variante de LSTTN que usa una semana, y qué precisión obtiene la arquitectura optimizada frente a los resultados publicados para PEMS08?

La comparación con el paper es **referencial**, no una reproducción: se conserva PEMS08, sus splits oficiales, horizontes y métricas, pero se reduce el contexto a una semana y se usa atención para la fusión.

## Candidatos

Solo cambia `d_model`, como en el análisis de hiperparámetros del paper:

| Candidato | d_model | Capas | Cabezas | Contexto | Patch | Máscara |
|---|---:|---:|---:|---:|---:|---:|
| MST-64 | 64 | 4 | 4 | 1 semana | 12 | 75 % |
| MST-96 | 96 | 4 | 4 | 1 semana | 12 | 75 % |

Ambos usan los mismos datos, semilla, decoder, optimizador, máximo de épocas y early stopping.

## Fases

1. Preentrenar ambos candidatos en paralelo.
2. Comparar la menor pérdida de reconstrucción en validación, tiempo y memoria.
3. Antes de la decisión definitiva, hacer un forecasting corto con ambos bajo los mismos parámetros y elegir principalmente por MAE de validación.
4. Congelar el ganador.
5. Ejecutar Optuna solo sobre los módulos de forecasting.
6. Entrenar el ganador con todo train, seleccionar por validación y consultar test una sola vez.
7. Reportar promedio global propio y horizontes de 15, 30 y 60 minutos.

## Ejecución en las dos RTX A6000

Crear el entorno e instalar primero un build moderno de PyTorch compatible con el driver del servidor; después:

```bash
pip install -r requirements-experiment.txt
```

Esta implementación no necesita `torch-geometric` ni depende de la carpeta ignorada `paper/`: incluye su propio módulo Graph WaveNet dentro de `lsttn_experiment/models`.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py pretrain --candidate mst_d64 --device cuda:0
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=1 python run_experiment.py pretrain --candidate mst_d96 --device cuda:0
```

Al usar `CUDA_VISIBLE_DEVICES`, cada proceso ve su GPU asignada como `cuda:0`.

Comparación:

```bash
python run_experiment.py compare
```

Forecasting corto de selección (sin consultar test), también en paralelo:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py probe --candidate mst_d64 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python run_experiment.py probe --candidate mst_d96 --device cuda:0
```

La decisión principal se toma con `best_valid_loss_normalized` de estos probes; la pérdida de reconstrucción, el tiempo y la memoria son criterios secundarios.

Optuna puede ejecutar dos workers sobre un mismo estudio. Repartir, por ejemplo, diez trials por proceso:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py tune \
  --checkpoint resultados_modular/pretraining/mst_d64/best.pt \
  --device cuda:0 --trials 10

CUDA_VISIBLE_DEVICES=1 python run_experiment.py tune \
  --checkpoint resultados_modular/pretraining/mst_d64/best.pt \
  --device cuda:0 --trials 10
```

Se debe reemplazar `mst_d64` por el candidato ganador.

Entrenamiento final:

```bash
CUDA_VISIBLE_DEVICES=0 python run_experiment.py train \
  --checkpoint resultados_modular/pretraining/mst_d64/best.pt \
  --params resultados_modular/optuna_best.json \
  --device cuda:0 --run-name propuesta_final
```

## Correcciones incorporadas

- Rutas relativas al proyecto.
- Splits oficiales `train_index.pkl`, `valid_index.pkl` y `test_index.pkl`.
- Dataset perezoso: las ventanas no se duplican en RAM.
- Inversa correcta de la normalización `[-1, 1]`.
- Decoder que mezcla tokens visibles y máscaras.
- Transformer congelado y en modo evaluación durante forecasting.
- Matrices de difusión reales del grafo, además de la adaptativa.
- Validación fija y test excluido de Optuna.
- `period_hidden` conectado realmente al modelo.
- Graph WaveNet como extractor corto, igual que la implementación concreta del paper.

## Comparación publicada para PEMS08

| Horizonte | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 15 min | 13.17 | 20.78 | 8.63 % |
| 30 min | 13.71 | 21.89 | 9.09 % |
| 60 min | 14.54 | 23.47 | 9.77 % |

El paper no publica un promedio global de los doce horizontes en esa tabla. Ese promedio se usa únicamente para comparar ejecuciones propias.

## Resultados y gráficas

Los artefactos se guardan dentro de `resultados_modular/`:

- `pretraining/<candidato>/learning_curve.png`: entrenamiento y validación del MST.
- `candidate_comparison.png`: comparación de reconstrucción y probes.
- `forecasting/probe_<candidato>/learning_curve.png`: curvas de selección del candidato.
- `optuna_trials.csv`: detalle reproducible de todos los trials.
- `optuna_plots/optuna_history.png`: evolución de la búsqueda.
- `optuna_plots/optuna_importance.png`: importancia de hiperparámetros cuando hay trials suficientes.
- `forecasting/<run>/learning_curve.png`: curva del entrenamiento final.
- `forecasting/<run>/test_metrics_by_horizon.png`: MAE, RMSE y MAPE de 5 a 60 minutos.
- `forecasting/<run>/prediction_example.png`: ejemplo real frente a pronosticado.
- `forecasting/<run>/test_predictions.npz`: predicciones y objetivos completos.

Los `metrics.json` son la fuente numérica principal. El comando `compare` reconstruye las curvas
de los preentrenamientos y probes desde esos archivos, incluso si fueron generados antes de
incorporar las gráficas. Por tanto, no es necesario repetir un entrenamiento ya terminado.
