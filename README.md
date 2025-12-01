# Memoria IBB Clustering

Versión modificada de Ibex con una estrategia de reinicio basada en clustering para el optimizador global. La idea es detectar estancamiento y reiniciar la exploración concentrando el esfuerzo en regiones prometedoras mediante una estrategia de reinicio adaptativo y el uso de clustering. Esta memoria sigue la línea de trabajo propuesta por Muñoz en https://github.com/Chelosky-O/Tesis_ibex_clustering.

## Descripción General

Los métodos de  Interval Branch-and-Bound pueden estancarse en regiones poco prometedoras. Este trabajo introduce reinicios guiados por clustering sobre el conjunto de celdas activas del buffer para acelerar la convergencia del optimizador global.

## Características

- Optimización global con Ibex (aritmética de intervalos).
- Reinicios por estancamiento con clustering:
  - K-Means++ (selección de centroides robusta).
  - HDBSCAN (clustering jerárquico por densidad, sin eps).
- Normalización de centros y filtro por volumen del hull.
- Ajuste dinámico de umbrales de reinicio según efectividad del reinicio.
- Estadísticas de reinicios y hulls creados.

## Estructura del Repositorio

```
Memoria-IBB-Clustering/
├── benchs/               # Ficheros de benchmark para pruebas
│   └── optim/
├── src/                  # Código fuente modificado de Ibex
│   └── optim/
│       ├── ibex_Optimizer.h
│       ├── ibex_Optimizer.cpp
│       └── ibex_DefaultOptimizerConfig.cpp
└── README.md            
```

## Requisitos

- Ibex (fuente) con módulo optim habilitado.
- Compilador C++17 o superior.

## Instalación y Compilación

Esta repo contiene archivos modificados que sustituyen a los de Ibex en src/optim a partir de las modificaciones de Muñoz.

1) Obtener Ibex:
```bash
git clone https://github.com/ibex-team/ibex-lib
cd ibex-lib
```

2) Hacer copia de seguridad de los ficheros originales:
```bash
cp src/optim/ibex_Optimizer.cpp src/optim/ibex_Optimizer.cpp.bak
cp src/optim/ibex_Optimizer.h   src/optim/ibex_Optimizer.h.bak
cp src/optim/ibex_DefaultOptimizerConfig.cpp src/optim/ibex_DefaultOptimizerConfig.cpp.bak
```

3) Copiar los archivos modificados desde esta repo:
```bash
# Ajusta las rutas a tu entorno
cp [repo-path]/src/optim/ibex_Optimizer.cpp                 [ibex-lib-path]/src/optim/ibex_Optimizer.cpp
cp [repo-path]/src/optim/ibex_Optimizer.h                   [ibex-lib-path]/src/optim/ibex_Optimizer.h  
cp [repo-path]/src/optim/ibex_DefaultOptimizerConfig.cpp    [ibex-lib-path]/src/optim/ibex_DefaultOptimizerConfig.cpp 
```

4) Compilar Ibex siguiendo los pasos del repo de Ibex: https://ibex-team.github.io/ibex-lib/

5) Probar con un benchmark de esta repo:
```bash
./bin/ibexopt [ibex-lib-path]/benchs/optim/medium/ex2_1_7.bch
```

## Uso y Configuración

Los parámetros del reinicio se establecen en el constructor del optimizador y durante la ejecución:

- Algoritmo de clustering:
  - Por defecto: K-Means++.
  - Alternativa: HDBSCAN.
- Umbrales de reinicio:
  - restart_threshold (iteraciones sin mejora).
  - node_threshold (tamaño del buffer).
- Filtro de volumen del hull:
  - hull_volume_threshold_fraction (fracción del volumen total del clúster).
- Normalización de centros:
  - use_normalization.

Cambios típicos (en el constructor del optimizador):
```cpp
// Cambia el algoritmo a HDBSCAN y ajusta parámetros
clustering_params.choice = ClusteringParams::Algorithm::HDBSCAN;
clustering_params.hdbscan_min_cluster_size = 15;   // tamaño mínimo
clustering_params.hdbscan_min_samples = -1;        // usa min_cluster_size

// Umbrales iniciales
restart_threshold = 500;  // iteraciones sin mejora
node_threshold    = 1000; // tamaño del buffer

// Normalización y filtro de volumen
clustering_params.use_normalization = true;
clustering_params.hull_volume_threshold_fraction = 3.0;
```

Notas prácticas:
- Los umbrales se penalizan/aumentan en función del nivel de estancamiento y la efectividad del reinicio (reducción del buffer y/o mejora de loup).
- El reinicio extrae el buffer, agrupa celdas, crea hulls por clúster y reinserta hulls aceptados o las celdas originales si el hull excede el umbral de volumen.

## Benchmarks

En `optim/` se incluyen problemas de prueba:
- medium/: ex2_1_7.bch, ex2_1_8.bch, ex2_1_9.bch, ex14_2_7.bch, etc.
- hard/, blowup/, coconutbenchmark-library2/, etc.

Ejemplo de ejecución:
```bash
./bin/ibexopt [ibex-lib-path]/benchs/optim/medium/ex14_2_7.bch
```