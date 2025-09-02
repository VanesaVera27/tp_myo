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

        capacidades = np.sum(self.S, axis=1)

        if self.A > 1 and np.ptp(capacidades) > 1e-9:
            n_clusters = min(self.A, len(np.unique(capacidades)))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = kmeans.fit_predict(capacidades.reshape(-1, 1))

            pasillos_por_grupo = [
                np.argmax(np.where(labels == g, capacidades, -np.inf))
                for g in range(n_clusters) if np.any(labels == g)
            ]
            pasillos_ordenados = sorted(pasillos_por_grupo,
                                        key=lambda a: capacidades[a],
                                        reverse=True)
        else:
            pasillos_ordenados = np.argsort(capacidades)[::-1].tolist()

        capacidades_ordenadas = capacidades[pasillos_ordenados]
        cap_acumuladas = np.cumsum(capacidades_ordenadas)

        lista_k_cap = sorted(enumerate(cap_acumuladas, start=1),
                             key=lambda x: x[1], reverse=True)
        lista_k = [k for k, _ in lista_k_cap]

        pesos = 1.0 / (np.arange(1, len(lista_k) + 1) ** alpha)
        pesos /= pesos.sum()
        lista_umbrales = (pesos * umbral).tolist()


        return lista_k, lista_umbrales
