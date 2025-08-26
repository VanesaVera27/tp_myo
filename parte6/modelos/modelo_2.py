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

    def Rankear(self, umbral, alpha=1.0, verbose=True):
        """
        Rankea valores de k en función de la capacidad acumulada.

        Parámetros
        ----------
        umbral : float
            Tiempo total disponible a repartir.
        alpha : float, default=1.0
            Factor de decaimiento de los pesos (1/(i+1)^alpha).
        verbose : bool, default=True
            Si True, imprime la asignación de tiempos.

        Retorna
        -------
        lista_k : list[int]
            Valores de k ordenados por prioridad.
        lista_umbrales : list[float]
            Tiempo asignado a cada k.
        """
        # --- Paso 1: calcular capacidades ---
        capacidades = np.sum(self.S, axis=1)
        n_clusters = min(4, self.A)

        # --- Paso 2: elegir pasillos representativos (clustering o sort simple) ---
        if self.A > 1 and np.ptp(capacidades) > 1e-9:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(capacidades.reshape(-1, 1))

            # elegir pasillo de mayor capacidad en cada cluster
            pasillos_por_grupo = [
                np.argmax(np.where(labels == g, capacidades, -np.inf))
                for g in range(n_clusters) if np.any(labels == g)
            ]
            pasillos_ordenados = sorted(pasillos_por_grupo, key=lambda a: capacidades[a], reverse=True)
        else:
            pasillos_ordenados = np.argsort(capacidades)[::-1].tolist()

        # --- Paso 3: capacidad acumulada para cada k ---
        capacidades_ordenadas = capacidades[pasillos_ordenados]
        cap_acumuladas = np.cumsum(capacidades_ordenadas)

        # ranking de k por capacidad acumulada
        lista_k_cap = sorted(enumerate(cap_acumuladas, start=1), key=lambda x: x[1], reverse=True)
        lista_k = [k for k, _ in lista_k_cap]

        # --- Paso 4: asignación de umbrales con pesos normalizados ---
        pesos = 1.0 / (np.arange(1, len(lista_k) + 1) ** alpha)
        pesos /= pesos.sum()
        lista_umbrales = (pesos * umbral).tolist()

        # --- Paso 5: logging ---
        if verbose:
            print("\n🕒 Asignación de tiempos por k (priorizando primeros):")
            for k, tiempo in zip(lista_k, lista_umbrales):
                print(f"  k = {k:<3} → {tiempo:.2f} s (peso {tiempo/umbral:.2%})")

        return lista_k, lista_umbrales
