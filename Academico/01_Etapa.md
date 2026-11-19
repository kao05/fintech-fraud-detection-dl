# Que se hará especificicamente en la primera etapa 


## Bloque A. Distribución y estructura de los datos

**Pregunta:** ¿Qué tan completa, diversa y estructuralmente consistente es nuestra información?

| Análisis | Visualización | Propósito |
| :--- | :--- | :--- |
| Valores faltantes por columna | Barplot horizontal | Identificar variables con mayor porcentaje de ausencia. |
| Distribución de valores faltantes | Histograma | Conocer cuántas columnas tienen determinados niveles de ausencia. |
| Cardinalidad de variables | Barplot horizontal | Identificar variables con pocos o muchos valores únicos. |
| Registros duplicados | Indicador o tabla | Identificar posibles duplicaciones. |
| Tipos de variables | Tabla resumen | Separar numéricas, categóricas, temporales e identificadores. |

* Variables con 2 o pocos valores únicos: posiblemente binarias o categóricas.
* Variables con cardinalidad moderada: posiblemente categóricas.
* Variables con cardinalidad muy elevada: posiblemente identificadores o características continuas.

## Bloque B. Distribución de la variable objetivo

**Pregunta:** ¿Qué tan desbalanceado está nuestro problema de clasificación?

Aquí ya contemplaste el gráfico de distribución de ``isFraud``, pero quiero que complementes esa visualización con tres medidas:

**Graficas**

1. Frecuencia absoluta de cada clase.
2. Frecuencia relativa o porcentaje de cada clase.
3. Ratio de desbalance entre la clase mayoritaria y la minoritaria.

## Bloque C. Análisis de importes de transacción

**Pregunta:** ¿El comportamiento económico de una transacción fraudulenta es diferente al de una legítima?

**Graficas**

1. Histograma de TransactionAmt separado por clase
2. Boxplot de importes por clase
3. Distribución de log1p(TransactionAmt) por clase

## Bloque D. Análisis temporal del fraude

**Pregunta:** ¿El comportamiento del fraude cambia con el tiempo?

**Graficas**

1. Evolución del volumen de transacciones
2. Evolución del número de fraudes
3. Tasa de fraude a través del tiempo

## Bloque E. Análisis de variables numéricas por clase

**Pregunta:** ¿Qué características numéricas presentan comportamientos diferentes entre operaciones legítimas y fraudulentas?

**Graficas**

1. Estadísticas descriptivas agrupadas por isFraud
   1. Media.
   2. Mediana.
   3. Desviación estándar.
   4. Mínimo y máximo.
   5. Percentiles 25, 50 y 75.

## Bloque F. Análisis de variables categóricas

**Pregunta:** ¿Existen categorías o características que presenten diferentes tasas de fraude?

**Graficas**

1. Frecuencia de las categorías (No necesitas representar todas las categorías si existen muchas; puedes comenzar con las 10 o 15 más frecuentes.)
2. Tasa de fraude por categoría
3. Asociación entre variables categóricas y el objetivo
    Puedes utilizar:
    - Chi-cuadrado de independencia.
    - V de Cramér como medida de asociación.


## Bloque G. Correlaciones y relaciones entre variables

**Pregunta:** ¿Qué variables presentan relaciones entre sí y cuáles muestran asociación con el fraude?

**Gráficas**

1. heatmap de correlaciones
2. Matriz de correlación entre variables numéricas
3. Asociación individual con isFraud
4. Matriz de correlación de las variables más relevantes

## Bloque H. Patrones de valores faltantes

**Pregunta:** ¿La ausencia de información puede ser un patrón relacionado con el fraude?

1. Crear indicadores de ausencia para las variables más interesantes.
2. Comparar el porcentaje de valores faltantes entre transacciones fraudulentas y legítimas.
3. Identificar variables cuya ausencia presente diferencias importantes entre clases.
4. Analizar si determinadas variables suelen estar ausentes simultáneamente.