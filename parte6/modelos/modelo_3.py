import sys
import os
import time
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

from parte5.columns_solver import construir_mejor_solucion

try:
    from parte5.columns_solver import Columns as ColumnsBase
except ImportError:
    # Añade '../parte5/' al sys.path si no se encuentra el módulo
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parte5.columns_solver import Columns as ColumnsBase

class Columns(ColumnsBase):
    def __init__(self, W, S, LB, UB):
        super().__init__(W, S, LB, UB)
        self.inactive_counter = {}  # {(k, id_col): [0,1,0,...]}
        self.iteracion_actual = {}  # {k: int}

    def actualizar_historial_inactividad(self, modelo_relajado, k):
        columnas_activas = self.columnas[k]
        for idx, x in enumerate(modelo_relajado.getVars()):
            columna_id = id(columnas_activas[idx])
            key = (k, columna_id)
            val = x.getLPSol() if x.getLPSol() is not None else 0

            if key not in self.inactive_counter:
                self.inactive_counter[key] = []

            self.inactive_counter[key].append(1 if val < 1e-5 else 0)

    def eliminar_columnas_inactivas_ultimas_iteraciones(self, k, umbral_iteraciones=5):
        columnas_activas = self.columnas[k]
        nuevas_columnas = []
        eliminadas = 0

        for col in columnas_activas:
            columna_id = id(col)
            key = (k, columna_id)
            historial = self.inactive_counter.get(key, [])

            if len(historial) >= umbral_iteraciones and all(v == 1 for v in historial[-umbral_iteraciones:]):
                eliminadas += 1
                continue
            nuevas_columnas.append(col)

        self.columnas[k] = nuevas_columnas
        if eliminadas > 0:
            print(f"🗑️ Eliminadas {eliminadas} columnas inactivas al final de k={k}")

    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        tiempo_inicializacion = 0.3 * umbral
        self.inicializar_columnas_para_k(k, umbral=tiempo_inicializacion)

        mejor_sol = None
        primera_iteracion = True

        while True:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ini
            tiempo_restante_total = umbral - tiempo_transcurrido
            if tiempo_restante_total <= 0:
                print("⏳ Tiempo agotado en Opt_cantidadPasillosFija → Fin del bucle.")
                break

            print(f"⌛ Iteración con {len(self.columnas[k])} columnas")

            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov = self.construir_modelo_maestro(k, tiempo_restante_total)
            maestro_relajado = Model(sourceModel=maestro)
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()
            maestro_relajado.relax()
            maestro_relajado.optimize()

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            if maestro_relajado.getStatus() == "optimal":
                pi = [maestro_relajado.getDualsolLinear(c) for c in maestro_relajado.getConss()]
            else:
                print("⚠️ No se encontró solución. Estado del modelo:", maestro_relajado.getStatus())
                return None

            if maestro_relajado.getStatus() in ["optimal", "feasible"]:
                valor_objetivo = maestro_relajado.getObjVal()
                print("Valor objetivo", valor_objetivo)
            else:
                print("⚠️ Modelo no óptimo ni factible.")
                return None

            mejor_sol = construir_mejor_solucion(maestro_relajado, self.columnas[k], valor_objetivo, self.cant_var_inicio)

            print("❤️ Cantidad de columnas antes de agregar:", len(self.columnas[k]))

            self.iteracion_actual[k] = self.iteracion_actual.get(k, 0) + 1
            self.actualizar_historial_inactividad(maestro_relajado, k)

            print("❤️ Cantidad de columnas después de historial:", len(self.columnas[k]))

            nueva_col = self.resolver_subproblema(self.W, self.S, pi, self.UB, k, tiempo_restante_total)
            if nueva_col is None:
                print("No se generó una columna mejoradora o era repetida → Fin del bucle.")
                break

            print("Nueva columna: ", nueva_col)
            print("➕ Columna nueva encontrada, se agrega.")
            self.agregar_columna(maestro, nueva_col, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov, k)
            print("💙 Cantidad de columnas después de agregar:", len(self.columnas[k]))

        iter_k = self.iteracion_actual.get(k, 0)
        if iter_k >= 5:
            self.eliminar_columnas_inactivas_ultimas_iteraciones(k, umbral_iteraciones=5)

        return mejor_sol

    def Opt_ExplorarCantidadPasillos(self, umbral):
        self.inactive_counter = {}
        self.iteracion_actual = {}
        self.columnas = {}
        best_sol = None
        tiempo_ini = time.time()

        lista_k, lista_umbrales = self.Rankear(umbral)

        for k, tiempo_k in zip(lista_k, lista_umbrales):
            tiempo_restante = umbral - (time.time() - tiempo_ini)
            if tiempo_restante <= 0:
                print("⏳ Sin tiempo restante para seguir evaluando k.")
                break

            print(f"Evaluando k={k} con tiempo asignado {tiempo_k:.2f} segundos")
            sol = self.Opt_cantidadPasillosFija(k, tiempo_k)

            if sol is not None:
                print("✅ Se encontró solución")
            else:
                print("❌ No se encontró una solución dentro del tiempo límite.")

            if sol:
                sol_obj = sol["valor_objetivo"]
                best_obj = best_sol["valor_objetivo"] if best_sol else -float('inf')
                if sol_obj > best_obj:
                    best_sol = sol

        if best_sol:
            tiempo_usado = time.time() - tiempo_ini
            tiempo_final = max(1.0, umbral - tiempo_usado)
            print("✅ Resultado final realajado:", best_sol)
            print(f"⏳ Tiempo restante para Opt_PasillosFijos: {tiempo_final:.2f}s")

            self.pasillos_fijos = best_sol["pasillos_seleccionados"]
            resultado_final = self.Opt_PasillosFijos(tiempo_final)

            if resultado_final is None:
                print("⚠️ Opt_PasillosFijos no devolvió una solución válida.")
                return {
                    "valor_objetivo": 0,
                    "ordenes_seleccionadas": set(),
                    "pasillos_seleccionados": set(),
                    "variables": best_sol.get("variables", 0),
                    "variables_final": best_sol.get("variables_final", 0),
                    "cota_dual": best_sol.get("cota_dual", 0)
                }

            resultado_final["tiempo_total"] = round(time.time() - tiempo_ini, 2)
            return resultado_final