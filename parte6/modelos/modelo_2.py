import numpy as np
from sklearn.cluster import KMeans
import sys
import os
try:
    from parte5.columns_solver import Columns as ColumnsBase
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parte5.columns_solver import Columns as ColumnsBase

class Columns(ColumnsBase):

    def __init__(self, W, S, LB, UB):
        super().__init__(W, S, LB, UB)
    
    def Rankear(self, umbral):
        capacidades = np.array([sum(self.S[a]) for a in range(self.A)]).reshape(-1, 1)
        n_clusters = min(4, self.A)

        if self.A > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(capacidades)
            labels = kmeans.labels_
            pasillos_por_grupo = []
            for g in range(n_clusters):
                indices = [i for i, lbl in enumerate(labels) if lbl == g]
                if indices:
                    mejor = max(indices, key=lambda a: capacidades[a][0])
                    pasillos_por_grupo.append(mejor)

            pasillos_ordenados = sorted(pasillos_por_grupo, key=lambda a: capacidades[a][0], reverse=True)
        else:
            pasillos_ordenados = [0]

        # Calcular capacidad acumulada por cada k (1 a A)
        lista_k_cap = []
        for k in range(1, self.A + 1):
            suma_capacidad_k = sum(capacidades[a][0] for a in pasillos_ordenados[:min(k, len(pasillos_ordenados))])
            lista_k_cap.append((k, suma_capacidad_k))

        # Ordenar por capacidad acumulada (mayor prioridad primero)
        lista_k_cap.sort(key=lambda x: x[1], reverse=True)
        lista_k = [k for k, _ in lista_k_cap]

        # Asignar pesos decrecientes del tipo 1, 1/2, 1/3, ..., normalizados
        pesos = [1 / (i + 1) for i in range(len(lista_k))]
        total_pesos = sum(pesos)
        lista_umbrales = [(p / total_pesos) * umbral for p in pesos]

        # Mostrar asignación
        print("\n🕒 Asignación de tiempos por k (priorizando primeros):")
        for k, tiempo in zip(lista_k, lista_umbrales):
            print(f"  k = {k:<3} → tiempo asignado: {tiempo:.2f} s")

        return lista_k, lista_umbrales

