<div align="center">
  <h1>Trabajo Práctico - Modelado y Optimización</h1>
  <h2>Universidad Nacional General Sarmiento</h2>
  <h3>Licenciatura en Sistemas</h3>
  <br>
  <strong>Integrantes:</strong><br>
  Juan Manuel Losada<br>
  Facundo Ruiz<br>
  Matías Morales<br>
  Vanesa Vera
  <br><br>
</div>

---

## Descripción General

Este trabajo práctico está dividido en dos grandes bloques:

- **Entrada en calor:** resolución de dos problemas clásicos de optimización combinatoria mediante programación lineal entera.
- **Desafío:** modelado y resolución de un problema complejo utilizando técnicas como descomposición y generación de columnas.

---

## Entrada en Calor

### Problema 1

El primer problema consiste en encontrar un contenedor $a \in A$ y un subconjunto de bolsitas $O' \subseteq O$ tales que:

1. La cantidad de veces que aparece cada ítem $i \in I$ en el contenedor alcanza para cubrir las apariciones de $i$ en $O'$.
2. Se maximice la suma de beneficios del contenedor y de los ítems de las bolsitas $O'$.

### Problema 2

El segundo problema consiste en encontrar subconjuntos $A' \subseteq A$ de contenedores y $O' \subseteq O$ de bolsitas tales que:

1. La cantidad de veces que aparece cada ítem $i \in I$ en $A' alcanza para cubrir las apariciones de $i$ en $O'$.
2. Se maximice la suma de beneficios de los contenedores de $A' y de los ítems de las bolsitas de $O'$.

---

## Desafío

El desafío consiste en resolver un problema de asignación de pasillos a órdenes para maximizar la cobertura, con restricciones sobre la cantidad de pasillos permitidos.


### Primera parte: Cantidad fija de pasillos

Modelo que, dado un número máximo \( M \) de pasillos, selecciona órdenes y pasillos para maximizar la cobertura.

**Archivo:** `parte1.py`


### Segunda parte: Pasillos fijos

Dado un conjunto de pasillos ya seleccionados, se eligen las órdenes que mejor se ajusten.

**Archivo:** `parte2.py`


### Tercera parte: Generación de columnas

Se reformula el problema como un modelo maestro restringido, y se agregan variables (columnas) dinámicamente con un subproblema de pricing.

**Archivo:** `parte3.py`  


### Cuarta parte: Exploración iterativa

Combina modelos de la primera y segunda parte. Se utiliza `Opt_cantidadPasillosFija` (modelo maestro) y `Opt_PasillosFijos` (submodelo) de forma iterativa, variando el número de pasillos \( k \), buscando mejorar los resultados globales.

**Función:** `Opt_ExplorarCantidadPasillos`  
**Archivo:** `parte4.py`

---

## Requisitos

Para ejecutar este proyecto necesitarás:

- Python 3.7 o superior
- Biblioteca PuLP para resolver problemas de programación lineal

## Instalación

1. Clone o descargue este repositorio
2. Instale la biblioteca PuLP:

```bash
pip install pulp
```

## Ejecución

Para ejecutar el programa con los datos de prueba:

### Para la entrada en calor:
```bash
python .\entrada_en_calor\main.py
```

### Para la entrada el desafio:
```bash
python .\desafio\part1.py
python .\desafio\part2.py
python .\desafio\part3.py
python .\desafio\part4.py
python .\desafio\part5.py
```


## Implementación

Los modelos de programación lineal se implementan utilizando PuLP, una biblioteca de Python para resolver problemas de optimización lineal. Cada variable de decisión es binaria, indicando si un contenedor o bolsita es seleccionado o no.

Las restricciones principales garantizan que la cantidad de cada tipo de ítem disponible en los contenedores seleccionados sea suficiente para cubrir la demanda de las bolsitas seleccionadas.

La función objetivo maximiza el beneficio total (suma de ítems en contenedores seleccionados y bolsitas seleccionadas).
