import sys
import os
import time
try:
    from parte5.columns_solver import Columns as ColumnsBase
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parte5.columns_solver import Columns as ColumnsBase

class Columns(ColumnsBase):
    def inicializar_columnas_para_k(self, k, umbral=None):
        tiempo_ini = time.time()

        if not hasattr(self, 'columnas'):
            self.columnas = {}

        self.columnas[k] = []
        unidades_o = [sum(self.W[o]) for o in range(self.O)]
        columnas_creadas = 0

        for a in range(self.A):
            if umbral and (time.time() - tiempo_ini) > umbral:
                break

            cap_restante = list(self.S[a])
            sel = [0] * self.O
            total_unidades = 0

            for o in range(self.O):
                if all(self.W[o][i] <= cap_restante[i] for i in range(self.I)) and \
                (total_unidades + unidades_o[o] <= self.UB):
                    sel[o] = 1
                    total_unidades += unidades_o[o]
                    for i in range(self.I):
                        cap_restante[i] -= self.W[o][i]

            if total_unidades > 0:
                self.columnas[k].append({
                    'pasillo': a,
                    'ordenes': sel,
                    'unidades': total_unidades
                })
                columnas_creadas += 1

