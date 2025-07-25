import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

import time
try:
    from parte5.columns_solver import Columns as ColumnsBase
    from parte5.columns_solver import construir_mejor_solucion

except ImportError:
    # Añade '../parte5/' al sys.path si no se encuentra el módulo
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from parte5.columns_solver import Columns as ColumnsBase
    from parte5.columns_solver import construir_mejor_solucion


class Columns(ColumnsBase):
    def __init__(self, W, S, LB, UB):
        super().__init__(W, S, LB, UB)  # Llama al init de ColumnsBase
        self.modelos = {}  # Inicializo el dict para guardar modelos por k
        self.n_pasillos = self.A
        self.inactive_counter = {}
        self.iteracion_actual = {}

    def inicializar_columnas_para_k(self, k, umbral=None):
        tiempo_ini = time.time()

        if not hasattr(self, 'columnas'):
            self.columnas = {}

        self.columnas[k] = []
        unidades_o = [sum(self.W[o]) for o in range(self.O)]
        columnas_creadas = 0

        for a in range(self.A):
            if umbral and (time.time() - tiempo_ini) > umbral:
                print("⏱️ Tiempo agotado durante inicialización de columnas")
                break

            cap_restante = list(self.S[a])
            sel = [0] * self.O
            total_unidades = 0

            # COLUMNA MAXIMAL para este pasillo
            for o in range(self.O):
                if all(self.W[o][i] <= cap_restante[i] for i in range(self.I)) and \
                (total_unidades + unidades_o[o] <= self.UB):
                    sel[o] = 1
                    total_unidades += unidades_o[o]
                    for i in range(self.I):
                        cap_restante[i] -= self.W[o][i]

            # Solo agregar la columna si se pudo cubrir al menos una orden
            if total_unidades > 0:
                self.columnas[k].append({
                    'pasillo': a,
                    'ordenes': sel,
                    'unidades': total_unidades
                })
                columnas_creadas += 1

        print(f"✅ {columnas_creadas} columnas iniciales maximales creadas para k = {k}")

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

    def actualizar_historial_inactividad(self, modelo_relajado, k, iteraciones_realizadas, umbral_iteraciones=5):
        columnas_activas = self.columnas[k]

        for idx, x in enumerate(modelo_relajado.getVars()):
            columna_id = id(columnas_activas[idx])
            key = (k, columna_id)
            val = x.getLPSol() if x.getLPSol() is not None else 0

            if key not in self.inactive_counter:
                self.inactive_counter[key] = []

            self.inactive_counter[key].append(1 if val < 1e-5 else 0)

            if len(self.inactive_counter[key]) > umbral_iteraciones:
                self.inactive_counter[key].pop(0)

    def eliminar_columnas_inactivas(self, k, iteraciones_realizadas, umbral_iteraciones=5, max_eliminar=None):
        columnas_activas = self.columnas[k]
        nuevas_columnas = []
        eliminadas = 0

        for idx, col in enumerate(columnas_activas):
            columna_id = id(col)
            key = (k, columna_id)

            historial = self.inactive_counter.get(key, [])
            n = min(iteraciones_realizadas, umbral_iteraciones)

            # Eliminar solo si estuvo inactiva en todas las últimas n iteraciones
            if len(historial) >= n and all(v == 1 for v in historial[-n:]):
                if max_eliminar is None or eliminadas < max_eliminar:
                    eliminadas += 1
                    continue  # No agregar esta columna
            nuevas_columnas.append(col)

        if eliminadas > 0:
            print(f"🗑️ Eliminadas {eliminadas} columnas inactivas en k={k} tras {iteraciones_realizadas} iteraciones")

        self.columnas[k] = nuevas_columnas


    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        
        # Reservar 30% del tiempo para inicialización
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

            # Construyo el modelo maestro
            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov = self.construir_modelo_maestro(k, tiempo_restante_total)

            # Copia del modelo maestro
            maestro_relajado = Model(sourceModel=maestro)

            # Desctivo para obtener los duales y restricciones correctamente
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()

            # Relajo el modelo maestro
            maestro_relajado.relax()
            maestro_relajado.optimize()

            # Obtener la cant de variables del 1er modelo maestro
            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            # Obtener los duales
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

            # Guardo la mejor solucion hasta el momento
            mejor_sol = construir_mejor_solucion(maestro_relajado, self.columnas[k], valor_objetivo, self.cant_var_inicio)

            print("❤️ Cantidad de columnas antes de agregar:", len(self.columnas[k]))

            # 🔁 Contador de iteraciones por k
            self.iteracion_actual[k] = self.iteracion_actual.get(k, 0) + 1
            iter_k = self.iteracion_actual[k]
            # Actualizo el historial de inactividad
            self.actualizar_historial_inactividad(maestro_relajado, k, iteraciones_realizadas=iter_k, umbral_iteraciones=5)

            print("❤️ Cantidad de columnas despues de eliminar y antes de agregar:", len(self.columnas[k]))

            # Subproblema que busca una columna mejoradora
            nueva_col = self.resolver_subproblema(self.W, self.S, pi, self.UB, k, tiempo_restante_total)

            if nueva_col is None:
                print("No se generó una columna mejoradora o era repetida → Fin del bucle.")
                break

            print("Nueva columna: ", nueva_col)
            print("➕ Columna nueva encontrada, se agrega.")
            self.agregar_columna(maestro, nueva_col, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov, k)
            print("💙 Cantidad de columnas después de agregar:", len(self.columnas[k]))
        
        # Después de salir del bucle, eliminar columnas inactivas
        iteraciones_realizadas = self.iteracion_actual.get(k, 0)
        if iteraciones_realizadas >= 5:
            self.eliminar_columnas_inactivas(k, iteraciones_realizadas=iteraciones_realizadas, umbral_iteraciones=5, max_eliminar=5)

        return mejor_sol
    
    def Opt_ExplorarCantidadPasillos(self, umbral):

        self.inactive_counter = {}
        self.iteracion_actual = {}

        self.columnas = {}
        best_sol = None
        tiempo_ini = time.time()

        # Nueva forma de llamar a Rankear
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

            print("✅ Resultado final con pasillos fijos:", resultado_final)
            print("✅ Cantidad de variables:", resultado_final["variables"])
            print("✅ Cantidad de variables finales:", resultado_final["variables_final"])

            resultado_final["tiempo_total"] = round(time.time() - tiempo_ini, 2)
            return resultado_final
