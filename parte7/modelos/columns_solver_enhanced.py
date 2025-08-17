import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

import time
try:
    from parte5.columns_solver import Columns as ColumnsBase

    from parte5.columns_solver import tiempo_excedido

except ImportError:
    # Añade '../parte5/' al sys.path si no se encuentra el módulo
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from parte5.columns_solver import Columns as ColumnsBase

    from parte5.columns_solver import tiempo_excedido


def construir_mejor_solucion(modelo_relajado, columnas_k, valor_objetivo, cant_var_inicio):
    """
    Construye la mejor solución a partir del modelo relajado y las columnas actuales.
    Compatible con columnas multi-pasillo.
    """
    mejor_sol = {
        'valor_objetivo': valor_objetivo,
        'columnas_seleccionadas': [],
        'pasillos_seleccionados': set(),
        'variables': cant_var_inicio  # <-- aquí guardamos la cantidad de columnas iniciales
    }

    # Iterar sobre las columnas
    for idx, col in enumerate(columnas_k):
        var_val = modelo_relajado.getVal(modelo_relajado.getVars()[idx])
        if var_val > 0.5:  # columna seleccionada
            mejor_sol['columnas_seleccionadas'].append(col)

            pasillos = col['pasillo']
            if isinstance(pasillos, int):
                pasillos = [pasillos]

            for p in pasillos:
                mejor_sol['pasillos_seleccionados'].add(p)

    mejor_sol['num_columnas'] = len(mejor_sol['columnas_seleccionadas'])

    return mejor_sol



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


#para no limitarse a un solo pasillo

    def construir_modelo_maestro(self, k, umbral):
        tiempo_ini = time.time()

        modelo = Model(f"RMP_k_{k}")
        modelo.setParam('display/verblevel', 0)
        x_vars = []

        # Crear variables binarias para cada columna
        for idx, col in enumerate(self.columnas[k]):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de variables.")
                return None, None, None, None, None, None

            pasillos = col['pasillo']
            if isinstance(pasillos, int):
                pasillos = [pasillos]

            x = modelo.addVar(vtype="B", name=f"x_{'_'.join(map(str, pasillos))}_{idx}")
            x_vars.append(x)

        # Restricción: seleccionar exactamente k columnas
        restr_card_k = modelo.addCons(quicksum(x_vars) == k, name="card_k")

        # Restricciones por órdenes
        restr_ordenes = {}
        for o in range(self.O):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de restricciones de órdenes.")
                return None, None, None, None, None, None

            cons = modelo.addCons(
                quicksum(x_vars[j] * self.columnas[k][j]['ordenes'][o] for j in range(len(x_vars))) <= 1,
                name=f"orden_{o}"
            )
            restr_ordenes[o] = cons

        # Restricción de unidades totales
        restr_ub = modelo.addCons(
            quicksum(x_vars[j] * self.columnas[k][j]['unidades'] for j in range(len(x_vars))) <= self.UB,
            name="restr_total_ub"
        )

        # Restricciones por pasillos
        restr_pasillos = {}
        for a in range(self.A):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de restricciones de pasillos.")
                return None, None, None, None, None, None

            cons = modelo.addCons(
                quicksum(
                    x_vars[j] for j in range(len(x_vars))
                    if a in (self.columnas[k][j]['pasillo'] if isinstance(self.columnas[k][j]['pasillo'], list) else [self.columnas[k][j]['pasillo']])
                ) <= 3,
                name=f"pasillo_{a}"
            )
            restr_pasillos[a] = cons

        # Función objetivo
        modelo.setObjective(
            quicksum(
                x_vars[j] * sum(
                    self.W[o][i]
                    for o in range(self.O) if self.columnas[k][j]['ordenes'][o]
                    for i in range(self.I)
                )
                for j in range(len(x_vars))
            ),
            sense="maximize"
        )

        return modelo, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_pasillos


    def resolver_subproblema(self, W, S, dual_vals, UB, k, umbral=None):
        tiempo_ini = time.time()

        O = len(W)
        I = len(W[0])
        A = len(S)

        if tiempo_excedido(tiempo_ini, umbral):
            return None

        units_o = [sum(W[o][i] for i in range(I)) for o in range(O)]

        pi_card_k = dual_vals["card_k"]
        pi_ordenes = [dual_vals[f"orden_{o}"] for o in range(O)]
        pi_ub = dual_vals["restr_total_ub"]
        pi_pasillos = [dual_vals[f"pasillo_{a}"] for a in range(A)]

        modelo = Model("Subproblema_multi_pasillo")
        modelo.setParam("display/verblevel", 0)

        # Variables binaras
        y = {a: modelo.addVar(vtype="B", name=f"y_{a}") for a in range(A)}
        z = {o: modelo.addVar(vtype="B", name=f"z_{o}") for o in range(O)}

        # Limitar máximo de pasillos por columna si se desea
        max_pasillos_por_columna = 3
        modelo.addCons(quicksum(y[a] for a in range(A)) <= max_pasillos_por_columna, name="max_pasillos")

        # Restricciones de capacidad por ítem
        for i in range(I):
            modelo.addCons(
                quicksum(W[o][i] * z[o] for o in range(O)) <=
                quicksum(S[a][i] * y[a] for a in range(A)),
                name=f"capacidad_total_item_{i}"
            )

        # Restricción de unidades totales
        modelo.addCons(quicksum(units_o[o] * z[o] for o in range(O)) <= UB, name="limite_unidades")

        # Función objetivo con costos duales
        modelo.setObjective(
            quicksum(units_o[o] * z[o] for o in range(O))
            - pi_card_k
            - quicksum(pi_ordenes[o] * z[o] for o in range(O))
            - pi_ub * quicksum(units_o[o] * z[o] for o in range(O))
            - quicksum(pi_pasillos[a] * y[a] for a in range(A)),
            sense="maximize"
        )

        modelo.optimize()

        if modelo.getStatus() != "optimal":
            return None

        reduced_cost = modelo.getObjVal()
        if reduced_cost <= 1e-6:
            return None

        pasillos_seleccionados = [a for a in range(A) if modelo.getVal(y[a]) > 0.5]
        ordenes = [int(modelo.getVal(z[o]) + 0.5) for o in range(O)]
        unidades = sum(units_o[o] for o in range(O) if ordenes[o])

        columna = {'pasillo': pasillos_seleccionados, 'ordenes': ordenes, 'unidades': unidades}

        return columna


    def Opt_PasillosFijos(self, umbral):
        tiempo_ini = time.time()
        k = len(self.pasillos_fijos)
        solucion_vacia = {"valor_objetivo": 0,"pasillos_seleccionados": set(),"ordenes_seleccionadas": set(),"restricciones": 0,"variables": 0,"variables_final": 0,"cota_dual": 0}
        
        # Calcular tiempo restante correctamente (usamos tiempo_ini para referencia)
        tiempo_restante_final = umbral - (time.time() - tiempo_ini)
        if tiempo_restante_final <= 0:
            print("⏳ No queda tiempo para Opt_PasillosFijos")
            return solucion_vacia

        # Validar que haya columnas para k
        if k not in self.columnas or not self.columnas[k]:
            print(f"❌ No hay columnas generadas para k = {k}")
            return solucion_vacia

        # Construir el modelo maestro con las columnas actuales
        modelo, x_vars, _, _, _, _ = self.construir_modelo_maestro(k, tiempo_restante_final)

        if modelo is None:
            print("❌ No se pudo construir el modelo maestro en Opt_PasillosFijos → tiempo agotado o error.")
            return solucion_vacia
        
        modelo.setPresolve(SCIP_PARAMSETTING.OFF)
        modelo.setHeuristics(SCIP_PARAMSETTING.OFF)
        modelo.disablePropagation()
        modelo.optimize()

        status = modelo.getStatus()

        if status in ["optimal", "feasible"] and modelo.getNSols() > 0:
            obj_val = modelo.getObjVal()
            pasillos_seleccionados = set()
            ordenes_seleccionadas = set()

            for idx, x in enumerate(x_vars):
                val = x.getLPSol()
                if val and val > 1e-5:
                    pasillos = self.columnas[k][idx]['pasillo']
                    if isinstance(pasillos, int):
                        pasillos = [pasillos]

                    for p in pasillos:
                        pasillos_seleccionados.add(p)

                    for o, seleccionado in enumerate(self.columnas[k][idx]['ordenes']):
                        if seleccionado:
                            ordenes_seleccionadas.add(o)

            mejor_sol = {
                "valor_objetivo": obj_val / len(pasillos_seleccionados) if pasillos_seleccionados else 0,
                "pasillos_seleccionados": pasillos_seleccionados,
                "ordenes_seleccionadas": ordenes_seleccionadas,
                "restricciones": modelo.getNConss(),
                "variables": 0,
                "variables_final": modelo.getNVars(),
                "cota_dual": modelo.getDualbound()
            }
        else:
            print(f"⚠️ Modelo no óptimo ni factible. Estado: {status}")
            mejor_sol = {
                "valor_objetivo": 0,
                "pasillos_seleccionados": set(),
                "ordenes_seleccionadas": set(),
                "restricciones": modelo.getNConss() if modelo else 0,
                "variables": 0,
                "variables_final": modelo.getNVars() if modelo else 0,
                "cota_dual": modelo.getDualbound() if modelo else 0
            }

        return mejor_sol
    
    def Opt_ExplorarCantidadPasillos(self, umbral):
        self.columnas = {}
        best_sol = None
        tiempo_ini = time.time()

        # Obtener lista de valores k y tiempos asignados a cada uno según Rankear()
        lista_k, lista_umbrales = self.Rankear(umbral)

        for k, tiempo_k in zip(lista_k, lista_umbrales):
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ini
            tiempo_restante_total = umbral - 4 - tiempo_transcurrido

            if tiempo_restante_total <= 0:
                print("⏳ Sin tiempo restante para seguir evaluando k.")
                break

            tiempo_k_deseado = tiempo_k
            tiempo_k = min(tiempo_k_deseado, tiempo_restante_total)

            print(f"Evaluando k={k} con tiempo asignado {tiempo_k:.2f} segundos")

            sol = self.Opt_cantidadPasillosFija(k, tiempo_k)

            if sol is not None:
                print("✅ Se encontró solución")
            else:
                print("❌ No se encontró una solución dentro del tiempo límite.")

            if sol:
                sol_obj = sol.get("valor_objetivo", -float('inf'))
                best_obj = best_sol.get("valor_objetivo", -float('inf')) if best_sol else -float('inf')
                if sol_obj > best_obj:
                    best_sol = sol

        if best_sol:
            tiempo_usado = time.time() - tiempo_ini
            tiempo_final = max(1.0, umbral - tiempo_usado)
            print("✅ Resultado final relajado:", best_sol)
            print(f"⏳ Tiempo restante para Opt_PasillosFijos: {tiempo_final:.2f}s")

            self.pasillos_fijos = best_sol["pasillos_seleccionados"]
            resultado_final = self.Opt_PasillosFijos(tiempo_final)
            resultado_final["variables"] = best_sol["variables"]

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
        
        else:
            print("⚠️ No se encontró ninguna solución durante la exploración.")
            return {
                "valor_objetivo": 0,
                "ordenes_seleccionadas": set(),
                "pasillos_seleccionados": set(),
                "variables": 0,
                "variables_final": 0,
                "cota_dual": 0
            }