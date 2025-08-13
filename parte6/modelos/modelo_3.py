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

    def actualizar_historial_inactividad(self, modelo_relajado, k, tiempo_ini, umbral):
        # Verificar tiempo restante
        tiempo_restante = umbral - (time.time() - tiempo_ini)
        if tiempo_restante <= 0:
            print("⏱️ Tiempo agotado antes de actualizar historial de inactividad.")
            return False  # Señal de que no se ejecutó

        columnas_activas = self.columnas[k]
        for idx, x in enumerate(modelo_relajado.getVars()):
            columna_id = id(columnas_activas[idx])
            key = (k, columna_id)
            val = x.getLPSol() if x.getLPSol() is not None else 0

            if key not in self.inactive_counter:
                self.inactive_counter[key] = []

            self.inactive_counter[key].append(1 if val < 1e-5 else 0)
        
        return True  # Señal de que se ejecutó correctamente

    def eliminar_columnas_inactivas_ultimas_iteraciones(self, k, tiempo_ini, umbral, umbral_iteraciones=5):
        # Verificar tiempo restante
        tiempo_restante = umbral - (time.time() - tiempo_ini)
        if tiempo_restante <= 0:
            print("⏱️ Tiempo agotado antes de eliminar columnas inactivas.")
            return False

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
        
        return True

    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        
        # Reservar 30% del tiempo para inicializar columnas iniciales para k
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

            print(f"⌛ Iteración con {len(self.columnas.get(k, []))} columnas")

            # Construcción del modelo maestro con las columnas actuales
            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_pasillos = self.construir_modelo_maestro(k, umbral)

            if maestro is None:
                print("❌ No se pudo construir el modelo maestro a tiempo → Fin del bucle.")
                break

            # Crear una copia para relajación y obtención de duales
            maestro_relajado = Model(sourceModel=maestro)

            # Desactivar ciertas heurísticas y preprocesos para obtener duales confiables
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()

            # Relajar modelo maestro (variables binarias a continuas)
            for var in maestro_relajado.getVars():
                maestro_relajado.chgVarType(var, "CONTINUOUS")
            
            maestro_relajado.optimize()

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            if maestro_relajado.getStatus() == "optimal":
                dual_map = {}
                dual_map = {cons.name: maestro_relajado.getDualSolVal(cons) for cons in maestro_relajado.getConss()}
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

            self.iteracion_actual[k] = self.iteracion_actual.get(k, 0) + 1
            self.actualizar_historial_inactividad(maestro_relajado, k, tiempo_ini, umbral)

            nueva_col = self.resolver_subproblema(self.W, self.S, dual_map, self.UB, k, tiempo_restante_total)
            if nueva_col is None:
                print("No se generó una columna mejoradora o era repetida → Fin del bucle.")
                break

            # Agregar nueva columna
            print("Nueva columna encontrada:", nueva_col)
            print("➕ Agregando columna nueva al modelo maestro.")
            self.columnas.setdefault(k, []).append(nueva_col)


        iter_k = self.iteracion_actual.get(k, 0)
        if iter_k >= 5:
            self.eliminar_columnas_inactivas_ultimas_iteraciones(k, tiempo_ini, umbral, umbral_iteraciones=5)

        return mejor_sol

