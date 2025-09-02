import sys
import os
import time
from collections import deque, defaultdict
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

from parte5.columns_solver import construir_mejor_solucion

try:
    from parte5.columns_solver import Columns as ColumnsBase
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parte5.columns_solver import Columns as ColumnsBase


class Columns(ColumnsBase):
    def __init__(self, W, S, LB, UB, verbose=True):
        super().__init__(W, S, LB, UB)
        self.inactive_counter = defaultdict(lambda: deque(maxlen=5))
        self.iteracion_actual = defaultdict(int)
        self.col_hashes = defaultdict(set)


    def tiempo_restante(self, tiempo_ini, umbral):
        return umbral - (time.time() - tiempo_ini)


    def actualizar_historial_inactividad(self, modelo_relajado, k, umbral_iteraciones=5):
        for idx, x in enumerate(modelo_relajado.getVars()):
            col = self.columnas[k][idx]
            key = (k, id(col))
            val = x.getLPSol() or 0
            self.inactive_counter[key].append(1 if val < 1e-5 else 0)

    def eliminar_columnas_inactivas(self, k, umbral_iteraciones=5):
        nuevas_columnas, eliminadas = [], 0
        for col in self.columnas[k]:
            key = (k, id(col))
            historial = self.inactive_counter.get(key, [])
            if len(historial) == umbral_iteraciones and all(historial):
                eliminadas += 1
            else:
                nuevas_columnas.append(col)

        self.columnas[k] = nuevas_columnas


    def _resolver_maestro_relajado(self, maestro):
        maestro_relajado = Model(sourceModel=maestro)
        maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
        maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
        maestro_relajado.disablePropagation()
        for var in maestro_relajado.getVars():
            maestro_relajado.chgVarType(var, "CONTINUOUS")

        maestro_relajado.optimize()
        if maestro_relajado.getStatus() != "optimal":
            return None, None

        dual_map = {cons.name: maestro_relajado.getDualSolVal(cons) for cons in maestro_relajado.getConss()}
        return maestro_relajado, dual_map

    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        self.inicializar_columnas_para_k(k, umbral=0.3 * umbral)

        mejor_sol, primera_iteracion = None, True
        
        REDUCED_COST_TOL = 1e-6

        while True:
            tiempo_restante = self.tiempo_restante(tiempo_ini, umbral)
            if tiempo_restante <= 0:
                break

            maestro, _, _, _, _, _ = self.construir_modelo_maestro(k, tiempo_restante)
            if maestro is None:
                return None

            maestro_relajado, dual_map = self._resolver_maestro_relajado(maestro)
            if maestro_relajado is None:
                break

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            mejor_sol = construir_mejor_solucion(
                maestro_relajado, self.columnas.get(k, []),
                maestro_relajado.getObjVal(), self.cant_var_inicio
            )

            self.iteracion_actual[k] += 1
            self.actualizar_historial_inactividad(maestro_relajado, k)

            nueva_col = self.resolver_subproblema(
                self.W, self.S, dual_map, self.UB, k, tiempo_restante
            )

            if nueva_col is None :
                break
            
            self.columnas.setdefault(k, []).append(nueva_col)


        if self.iteracion_actual[k] >= 5:
            self.eliminar_columnas_inactivas(k)

        return mejor_sol