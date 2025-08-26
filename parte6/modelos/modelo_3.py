import sys
import os
import time
from collections import deque
from pyscipopt import Model, SCIP_PARAMSETTING

from parte5.columns_solver import construir_mejor_solucion

try:
    from parte5.columns_solver import Columns as ColumnsBase
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parte5.columns_solver import Columns as ColumnsBase


class Columns(ColumnsBase):
    def __init__(self, W, S, LB, UB):
        super().__init__(W, S, LB, UB)
        self.inactive_counter = {}   # {(k, id_col): deque([0,1,...])}
        self.iteracion_actual = {}   # {k: int}
        self.col_hashes = {}         # {k: set()}

    def tiempo_restante(self, tiempo_ini, umbral):
        return umbral - (time.time() - tiempo_ini)

    def actualizar_historial_inactividad(self, modelo_relajado, k, tiempo_ini, umbral, umbral_iteraciones=5):
        if self.tiempo_restante(tiempo_ini, umbral) <= 0:
            return False  

        for idx, x in enumerate(modelo_relajado.getVars()):
            col = self.columnas[k][idx]
            key = (k, id(col))
            val = x.getLPSol() or 0

            if key not in self.inactive_counter:
                self.inactive_counter[key] = deque(maxlen=umbral_iteraciones)

            self.inactive_counter[key].append(1 if val < 1e-5 else 0)
        
        return True

    def eliminar_columnas_inactivas_ultimas_iteraciones(self, k, umbral_iteraciones=5):
        nuevas_columnas = []
        eliminadas = 0

        for col in self.columnas[k]:
            key = (k, id(col))
            historial = self.inactive_counter.get(key, [])
            if len(historial) == umbral_iteraciones and all(historial):
                eliminadas += 1
                continue
            nuevas_columnas.append(col)

        self.columnas[k] = nuevas_columnas
        if eliminadas:
            print(f"🗑️ Eliminadas {eliminadas} columnas inactivas (k={k})")

    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        tiempo_inicializacion = 0.3 * umbral
        self.inicializar_columnas_para_k(k, umbral=tiempo_inicializacion)

        self.col_hashes.setdefault(k, set())

        mejor_sol = None
        primera_iteracion = True

        while True:
            if self.tiempo_restante(tiempo_ini, umbral) <= 0:
                print("⏳ Tiempo agotado en Opt_cantidadPasillosFija → Fin del bucle.")
                break

            print(f"⌛ Iteración con {len(self.columnas.get(k, []))} columnas")

            # --- Construir maestro ---
            maestro, _, restr_card_k, restr_ordenes, restr_ub, restr_pasillos = \
                self.construir_modelo_maestro(k, self.tiempo_restante(tiempo_ini, umbral))
            if maestro is None:
                print("No se pudo construir el modelo maestro a tiempo")
                return None

            # --- Relajar maestro ---
            maestro_relajado = Model(sourceModel=maestro)
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()
            for var in maestro_relajado.getVars():
                maestro_relajado.chgVarType(var, "CONTINUOUS")
            maestro_relajado.optimize()

            if maestro_relajado.getStatus() != "optimal":
                print("⚠️ No se encontró solución. Estado:", maestro_relajado.getStatus())
                break

            # --- Duales del maestro ---
            dual_map = {cons.name: maestro_relajado.getDualSolVal(cons) for cons in maestro_relajado.getConss()}

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            # --- Calcular reduced costs ---
            rc_columnas = []
            for j, col in enumerate(self.columnas.get(k, [])):
                # Precomputar costo si no existe
                if "costo" not in col:
                    col["costo"] = sum(
                        self.W[o][i]
                        for i in range(self.I)
                        for o in range(self.O)
                        if col["ordenes"][o]
                    )
                c_j = col["costo"]

                dual_contrib = dual_map.get("card_k", 0)
                dual_contrib += sum(col["ordenes"][o] * dual_map.get(f"orden_{o}", 0) for o in range(self.O))
                dual_contrib += col["unidades"] * dual_map.get("restr_total_ub", 0)
                dual_contrib += dual_map.get(f"pasillo_{col['pasillo']}", 0)

                rc = c_j - dual_contrib
                rc_columnas.append(rc)

            # --- Parada por optimalidad ---
            if all(rc >= -1e-6 for rc in rc_columnas):
                print("✅ No hay columnas mejoradoras → óptimo alcanzado.")
                break

            # --- Construir mejor solución ---
            mejor_sol = construir_mejor_solucion(
                maestro_relajado, self.columnas.get(k, []),
                maestro_relajado.getObjVal(), self.cant_var_inicio
            )

            # --- Actualizar iteración e historial ---
            self.iteracion_actual[k] = self.iteracion_actual.get(k, 0) + 1
            self.actualizar_historial_inactividad(maestro_relajado, k, tiempo_ini, umbral)

            # --- Generar nueva columna ---
            nueva_col = self.resolver_subproblema(self.W, self.S, dual_map, self.UB, k, self.tiempo_restante(tiempo_ini, umbral))
            if nueva_col is None:
                print("No se generó columna nueva → Fin del bucle.")
                break

            # Precomputar costo
            nueva_col["costo"] = sum(
                self.W[o][i]
                for i in range(self.I)
                for o in range(self.O)
                if nueva_col["ordenes"][o]
            )

            # Hash para duplicados
            col_hash = (nueva_col["pasillo"], tuple(nueva_col["ordenes"]))
            if col_hash in self.col_hashes[k]:
                print("Columna duplicada → Fin del bucle.")
                break

            # Agregar columna válida
            print("➕ Nueva columna encontrada")
            self.columnas.setdefault(k, []).append(nueva_col)
            self.col_hashes[k].add(col_hash)

        # --- Limpieza de columnas inactivas ---
        if self.iteracion_actual.get(k, 0) >= 5:
            self.eliminar_columnas_inactivas_ultimas_iteraciones(k, umbral_iteraciones=5)

        return mejor_sol
