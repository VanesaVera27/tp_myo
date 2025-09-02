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
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from parte5.columns_solver import Columns as ColumnsBase

    from parte5.columns_solver import tiempo_excedido

def tiempo_excedido(tiempo_ini, umbral):
    return time.time() - tiempo_ini > umbral

def construir_mejor_solucion(maestro_relajado, columnas_k, cant_var_inicio, k):
    pasillos_seleccionados = set()
    ordenes_seleccionadas = set()
    valor_obj_real = 0


    for idx, x in enumerate(maestro_relajado.getVars()):
        val = maestro_relajado.getVal(x)
        if val > 1e-6:  
            pasillos_seleccionados.update(columnas_k[idx].get('pasillos', []))
            for o, sel in enumerate(columnas_k[idx]['ordenes']):
                if sel:
                    ordenes_seleccionadas.add(o)
    valor_obj_real = maestro_relajado.getObjVal()

    cota_dual_real = maestro_relajado.getDualbound()
    gap_real = valor_obj_real - cota_dual_real

  
    productividad = valor_obj_real / k if k > 0 else 0

    mejor_sol = {
        "valor_objetivo_total": valor_obj_real,
        "productividad_por_pasillo": productividad,
        "pasillos_seleccionados": pasillos_seleccionados,
        "ordenes_seleccionadas": ordenes_seleccionadas,
        "variables": cant_var_inicio,
        "variables_final": maestro_relajado.getNVars(),
        "cota_dual": cota_dual_real,
        "gap_real": gap_real
    }

    print(f"🔹 Mejor solución construida: Total={valor_obj_real:.6f}, "
          f"Prod/ps={productividad:.6f}, Dual={cota_dual_real:.6f}, Gap={gap_real:.6e}")

    return mejor_sol


class Columns(ColumnsBase):
    def __init__(self, W, S, LB, UB):
        super().__init__(W, S, LB, UB) 
        self.modelos = {}  
        self.n_pasillos = self.A
        self.columnas_iniciales = []   
        self.columnas = {}  
        self.inactive_counter = {}
        self.iteracion_actual = {}

    def inicializar_columnas_iniciales(self, umbral=None):
            tiempo_ini = time.time()
            self.columnas_iniciales = []

            W_np = np.array(self.W) 
            S_np = np.array(self.S)  
            unidades_o = W_np.sum(axis=1)  
            ordenes_indexadas = np.argsort(-unidades_o) 
            

            for a in range(self.A):
                if umbral and (time.time() - tiempo_ini) > umbral:
                    break

                columna_generada = False

                cap_restante = S_np[a].copy()
                sel = np.zeros(self.O, dtype=int)
                total_unidades = 0

                for o in range(self.O):
                    if np.all(W_np[o] <= cap_restante) and (total_unidades + unidades_o[o] <= self.UB):
                        sel[o] = 1
                        total_unidades += unidades_o[o]
                        cap_restante -= W_np[o]

                if total_unidades > 0:
                    self.columnas_iniciales.append({'pasillos':[a], 'ordenes':sel.tolist(), 'unidades':total_unidades})
                    columna_generada = True

            if len(self.columnas_iniciales) == 0:
                for a in range(self.A):
                    for b in range(a + 1, self.A):
                        if umbral and (time.time() - tiempo_ini) > umbral:
                            break

                        cap_restante = S_np[a] + S_np[b]
                        sel = np.zeros(self.O, dtype=int)
                        total_unidades = 0

                        for o in ordenes_indexadas:
                            if unidades_o[o] + total_unidades <= self.UB and np.all(W_np[o] <= cap_restante):
                                sel[o] = 1
                                total_unidades += unidades_o[o]
                                cap_restante -= W_np[o]

                        if total_unidades > 0:
                            self.columnas_iniciales.append({'pasillos':[a,b], 'ordenes':sel.tolist(), 'unidades':total_unidades})
                            break 

            print(f"✅ {len(self.columnas_iniciales)} columnas iniciales generadas para todos los k posibles")
    
    def construir_modelo_maestro(self, k, umbral):
        tiempo_ini = time.time()

        modelo = Model(f"RMP_k_{k}")
        modelo.setParam('display/verblevel', 0)
        modelo.setParam('limits/time', umbral)
        x_vars = []

        for idx, col in enumerate(self.columnas[k]):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de variables.")
                return None, None, None, None, None, None

            x = modelo.addVar(vtype="B", name=f"x_{idx}")
            x_vars.append(x)

        restr_card_k = modelo.addCons(
            quicksum(x_vars[j] * len(self.columnas[k][j].get('pasillos', [])) for j in range(len(x_vars))) == k,
            name="card_k"
        )

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

        if tiempo_excedido(tiempo_ini, umbral):
            print("⏱️ Tiempo excedido antes de la restricción de unidades totales.")
            return None, None, None, None, None, None

        restr_ub = modelo.addCons(
            quicksum(x_vars[j] * self.columnas[k][j]['unidades'] for j in range(len(x_vars))) <= self.UB,
            name="restr_total_ub"
        )

        restr_pasillos = {}
        for a in range(self.A):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de restricciones de pasillos.")
                return None, None, None, None, None, None

            cons = modelo.addCons(
                quicksum(
                    x_vars[j] for j in range(len(x_vars)) if a in self.columnas[k][j].get('pasillos', [])
                ) <= 1,
                name=f"pasillo_{a}"
            )
            restr_pasillos[a] = cons

        if tiempo_excedido(tiempo_ini, umbral):
            print("⏱️ Tiempo excedido antes de definir la función objetivo.")
            return None, None, None, None, None, None

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
            print("⏱️ Tiempo excedido antes de comenzar.")
            return None

        units_o = [sum(W[o][i] for i in range(I)) for o in range(O)]

        modelo = Model("Subproblema_unico")
        modelo.setParam("display/verblevel", 0)
        modelo.setParam('limits/time', umbral)

        y = {a: modelo.addVar(vtype="B", name=f"y_{a}") for a in range(A)}
        z = {o: modelo.addVar(vtype="B", name=f"z_{o}") for o in range(O)}

        modelo.addCons(quicksum(y[a] for a in range(A)) == k, name="al_menos_un_pasillo")

        for i in range(I):
            modelo.addCons(
                quicksum(W[o][i] * z[o] for o in range(O)) <=
                quicksum(S[a][i] * y[a] for a in range(A)),
                name=f"capacidad_total_item_{i}"
            )

        modelo.addCons(quicksum(units_o[o] * z[o] for o in range(O)) <= UB,
                    name="limite_unidades")

        expr_cj = quicksum(units_o[o] * z[o] for o in range(O))

        expr_Ajy = dual_vals.get("card_k", 0) * quicksum(y[a] for a in range(A)) \
            + quicksum(dual_vals.get(f"orden_{o}", 0) * z[o] for o in range(O)) \
            + dual_vals.get("restr_total_ub", 0) * quicksum(units_o[o] * z[o] for o in range(O)) \
            + quicksum(dual_vals.get(f"pasillo_{a}", 0) * y[a] for a in range(A))

        modelo.setObjective(expr_cj - expr_Ajy, sense="maximize")
        modelo.optimize()

        if modelo.getStatus() != "optimal":
            print("⚠️ Subproblema no óptimo.")
            return None
        
        reduced_cost = modelo.getObjVal()
        print(f"🔹 Costo reducido subproblema: {reduced_cost:.6f}")

        if reduced_cost <= 1e-6:
            return None

        pasillos_seleccionados = [a for a in range(A) if modelo.getVal(y[a]) > 0.5]
        ordenes = [int(modelo.getVal(z[o]) + 0.5) for o in range(O)]
        unidades = sum(units_o[o] for o in range(O) if ordenes[o])

        return {'pasillos': pasillos_seleccionados, 'ordenes': ordenes, 'unidades': unidades}


    def Opt_cantidadPasillosFija(self, k, umbral):
        tiempo_ini = time.time()
        tiempo_inicializacion = 0.3 * umbral
        self.columnas[k] = self.columnas_iniciales.copy()

        mejor_sol_global = None
        mejor_prod_global = -1
        primera_iteracion = True

        while True:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ini
            tiempo_restante_total = umbral - tiempo_transcurrido

            if tiempo_restante_total <= 0:
                print("⏳ Tiempo agotado en Opt_cantidadPasillosFija → Fin del bucle.")
                break

            print(f"⌛ Iteración con {len(self.columnas.get(k, []))} columnas")

            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_pasillos = self.construir_modelo_maestro(k, tiempo_restante_total)
            if maestro is None:
                print("No se pudo construir el modelo maestro a tiempo")
                break

            maestro_relajado = Model(sourceModel=maestro)
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setParam('limits/time', tiempo_restante_total)
            maestro_relajado.disablePropagation()
            for var in maestro_relajado.getVars():
                maestro_relajado.chgVarType(var, "CONTINUOUS")
            maestro_relajado.optimize()

            if maestro_relajado.getStatus() != "optimal":
                print("⚠️ No se encontró solución. Estado del modelo:", maestro_relajado.getStatus())
                break

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            valor_objetivo_primal = maestro_relajado.getObjVal()
            sol_actual = construir_mejor_solucion(maestro_relajado, self.columnas.get(k, []),
                                                self.cant_var_inicio,k)

            if sol_actual['productividad_por_pasillo'] > mejor_prod_global:
                mejor_prod_global = sol_actual['productividad_por_pasillo']
                mejor_sol_global = sol_actual
                print("🎉🎉🎉 Mejor solución actual:", mejor_sol_global)

            dual_map = {cons.name: maestro_relajado.getDualSolVal(cons) for cons in maestro_relajado.getConss()}
            tiempo_restante_total = umbral - (time.time() - tiempo_ini)
            nueva_col = self.resolver_subproblema(self.W, self.S, dual_map, self.UB, k, tiempo_restante_total)

            if nueva_col is None:
                print("No se generó columna nueva o ya existe → Fin del bucle.")
                break

            print("Nueva columna encontrada:", nueva_col)
            self.columnas.setdefault(k, []).append(nueva_col)

        return mejor_sol_global


    def Opt_PasillosFijos(self, umbral):
        tiempo_ini = time.time()
        k = len(self.pasillos_fijos)
        solucion_vacia = {
            "valor_objetivo": 0,
            "productividad_por_pasillo": 0,
            "pasillos_seleccionados": set(),
            "ordenes_seleccionadas": set(),
            "restricciones": 0,
            "variables": 0,
            "variables_final": 0,
            "cota_dual": 0,
            "tiempo_total": 0
        }

        tiempo_restante_final = umbral - (time.time() - tiempo_ini)
        if tiempo_restante_final <= 0:
            print("⏳ No queda tiempo para Opt_PasillosFijos")
            return solucion_vacia

        if k not in self.columnas or not self.columnas[k]:
            print(f"❌ No hay columnas generadas para k = {k}")
            return solucion_vacia

        modelo, x_vars, _, _, _, _ = self.construir_modelo_maestro(k, tiempo_restante_final)
        if modelo is None:
            print("❌ No se pudo construir el modelo maestro en Opt_PasillosFijos → tiempo agotado o error.")
            return solucion_vacia

        modelo.setPresolve(SCIP_PARAMSETTING.OFF)
        modelo.setHeuristics(SCIP_PARAMSETTING.OFF)
        modelo.disablePropagation()
        modelo.optimize()

        status = modelo.getStatus()
        tiempo_total = time.time() - tiempo_ini

        if status in ["optimal", "feasible"] and modelo.getNSols() > 0:
            valor_obj_real = 0
            pasillos_seleccionados = set()
            ordenes_seleccionadas = set()

            for idx, x in enumerate(x_vars):
                val = x.getLPSol()
                if val and val > 1e-6:
                    pasillos_seleccionados.update(self.columnas[k][idx].get('pasillos', []))
                    for o, sel in enumerate(self.columnas[k][idx]['ordenes']):
                        if sel:
                            ordenes_seleccionadas.add(o)
                    valor_obj_real += self.columnas[k][idx]['unidades']
            
            mejor_sol = {
                "valor_objetivo":  valor_obj_real / k if k > 0 else 0,
                "productividad_por_pasillo": valor_obj_real / k if k > 0 else 0,
                "pasillos_seleccionados": pasillos_seleccionados,
                "ordenes_seleccionadas": ordenes_seleccionadas,
                "restricciones": modelo.getNConss(),
                "variables": 0,
                "variables_final": modelo.getNVars(),
                "cota_dual": modelo.getDualbound(),
                "tiempo_total": tiempo_total
            }
        else:
            print(f"⚠️ Modelo no óptimo ni factible. Estado: {status}")
            mejor_sol = solucion_vacia
            mejor_sol.update({
                "restricciones": modelo.getNConss() if modelo else 0,
                "variables_final": modelo.getNVars() if modelo else 0,
                "cota_dual": modelo.getDualbound() if modelo else 0,
                "tiempo_total": tiempo_total
            })

        return mejor_sol



    def Opt_ExplorarCantidadPasillos(self, umbral):
        self.columnas = {}
        best_sol = None
        tiempo_ini = time.time()

        ratio_inicial = 0.25
        tiempo_inicializacion = umbral * ratio_inicial
        tiempo_final_fijo = 60

        lista_k, lista_umbrales = self.Rankear(umbral)
        self.inicializar_columnas_iniciales(tiempo_inicializacion)

        for k, tiempo_k_estimado in zip(lista_k, lista_umbrales):
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ini
            tiempo_restante_total = umbral - tiempo_final_fijo - tiempo_transcurrido

            if tiempo_restante_total <= 0:
                print("⏳ Sin tiempo restante para seguir evaluando k.")
                break

            tiempo_k = min(tiempo_k_estimado, tiempo_restante_total)
            print(f"Evaluando k={k} con tiempo asignado {tiempo_k:.2f} segundos")

            sol = self.Opt_cantidadPasillosFija(k, tiempo_k)

            if sol:
                sol_prod = sol.get("productividad_por_pasillo", -float('inf'))
                best_prod = best_sol.get("productividad_por_pasillo", -float('inf')) if best_sol else -float('inf')
                if sol_prod > best_prod:
                    best_sol = sol

        if best_sol:
            print("✅ Mejor solución global encontrada:", best_sol)

            self.pasillos_fijos = best_sol["pasillos_seleccionados"]
            resultado_final = self.Opt_PasillosFijos(tiempo_final_fijo)
            resultado_final["tiempo_total"] = round(time.time() - tiempo_ini, 2)
            resultado_final["variables"] = best_sol["variables"]

            print("✅ Resultado final con pasillos fijos:", resultado_final)
            return resultado_final

        print("⚠️ No se encontró ninguna solución durante la exploración.")
        return {
            "valor_objetivo": 0,
            "ordenes_seleccionadas": set(),
            "pasillos_seleccionados": set(),
            "variables": 0,
            "variables_final": 0,
            "cota_dual": 0
        }

    
    def Rankear(self, umbral):

        lista_k = []
        max_chicos = min(self.A, 5)
        lista_k.extend(range(1, max_chicos + 1))

        if self.A > 6:
            lista_k.append(self.A // 2)

        if self.A not in lista_k:
            lista_k.append(self.A)

        lista_k = sorted(set(lista_k))

        tiempo_por_k = max(1.0, umbral / len(lista_k))
        lista_umbrales = [tiempo_por_k] * len(lista_k)

        return lista_k,lista_umbrales