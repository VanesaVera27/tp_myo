import time
import random
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

def tiempo_excedido(tiempo_ini, umbral):
    return time.time() - tiempo_ini > umbral

def construir_mejor_solucion(modelo_relajado, columnas_k, valor_obj_primal, cant_var_inicio):
    pasillos_seleccionados = set()
    ordenes_seleccionadas = set()

    for idx, x in enumerate(modelo_relajado.getVars()):
        val = modelo_relajado.getVal(x)
        if val and val > 1e-5:
            pasillos_seleccionados.add(columnas_k[idx]['pasillo'])
            for o, seleccionado in enumerate(columnas_k[idx]['ordenes']):
                if seleccionado:
                    ordenes_seleccionadas.add(o)

    cota_dual_real = modelo_relajado.getDualbound()
    gap_real = valor_obj_primal - cota_dual_real

    mejor_sol = {
        "valor_objetivo": valor_obj_primal / len(pasillos_seleccionados) if pasillos_seleccionados else 0,
        "pasillos_seleccionados": pasillos_seleccionados,
        "ordenes_seleccionadas": ordenes_seleccionadas,
        "variables": cant_var_inicio,
        "variables_final": modelo_relajado.getNVars(),
        "cota_dual": cota_dual_real,
        "gap_real": gap_real
    }


    return mejor_sol

class Columns:
    def __init__(self, W, S, LB, UB):
        self.W = W
        self.S = S
        self.LB = LB
        self.UB = UB
        self.O = len(W)
        self.I = len(W[0])
        self.A = len(S)
        self.columnas = {}
        self.pasillos_fijos = []
        self.cant_var_inicio = 0

    def inicializar_columnas_para_k(self, k, umbral=None):
        tiempo_ini = time.time()

        if not hasattr(self, 'columnas'):
            self.columnas = {}

        self.columnas[k] = []
        unidades_o = [sum(self.W[o]) for o in range(self.O)]

        ordenes_indexadas = list(range(self.O))
        random.shuffle(ordenes_indexadas)

        for a in range(self.A):
            if umbral and (time.time() - tiempo_ini) > umbral:
                print("⏱️ Tiempo agotado durante inicialización de columnas")
                break
            
            for _ in range(5): 
                cap_restante = list(self.S[a])
                sel = [0] * self.O
                total_unidades = 0
                for o in ordenes_indexadas:
                    if (unidades_o[o] + total_unidades <= self.UB) and \
                    all(self.W[o][i] <= cap_restante[i] for i in range(self.I)):
                        
                        sel[o] = 1
                        total_unidades += unidades_o[o]
                        for i in range(self.I):
                            cap_restante[i] -= self.W[o][i]
                
                if total_unidades > 0:
                    nueva_columna = {
                        'pasillo': a,
                        'ordenes': sel,
                        'unidades': total_unidades
                    }
                    self.columnas[k].append(nueva_columna)

        print(f"✅ {len(self.columnas[k])} columnas iniciales creadas para k = {k}")



    def construir_modelo_maestro(self, k, umbral):
        tiempo_ini = time.time()

        modelo = Model(f"RMP_k_{k}")
        modelo.setParam('display/verblevel', 0)
        x_vars = []

        for idx, col in enumerate(self.columnas[k]):
            if tiempo_excedido(tiempo_ini, umbral):
                print("⏱️ Tiempo excedido durante la creación de variables.")
                return None, None, None, None, None, None

            x = modelo.addVar(vtype="B", name=f"x_{col['pasillo']}_{idx}")
            x_vars.append(x)

        restr_card_k = modelo.addCons(quicksum(x_vars) == k, name="card_k")

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
                    x_vars[j] for j in range(len(x_vars)) if self.columnas[k][j]['pasillo'] == a
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

        y = {a: modelo.addVar(vtype="B", name=f"y_{a}") for a in range(A)}
        z = {o: modelo.addVar(vtype="B", name=f"z_{o}") for o in range(O)}

        modelo.addCons(quicksum(y[a] for a in range(A)) == 1, name="unico_pasillo")

        for i in range(I):
            modelo.addCons(
                quicksum(W[o][i] * z[o] for o in range(O)) <=
                quicksum(S[a][i] * y[a] for a in range(A)),
                name=f"capacidad_total_item_{i}"
            )

        modelo.addCons(quicksum(units_o[o] * z[o] for o in range(O)) <= UB,
                    name="limite_unidades")

        expr_cj = quicksum(units_o[o] * z[o] for o in range(O))
        expr_Ajy = dual_vals.get("card_k", 0) \
            + quicksum(dual_vals.get(f"orden_{o}", 0) * z[o] for o in range(O)) \
            + dual_vals.get("restr_total_ub", 0) * quicksum(units_o[o] * z[o] for o in range(O)) \
            + quicksum(dual_vals.get(f"pasillo_{a}", 0) * y[a] for a in range(A))

        modelo.setObjective(expr_cj - expr_Ajy, sense="maximize")
        modelo.optimize()

        if modelo.getStatus() != "optimal":
            print("⚠️ Subproblema no óptimo.")
            return None

        reduced_cost = modelo.getObjVal()

        if reduced_cost <= 1e-6:
            return None

        pasillo_seleccionado = next((a for a in range(A) if modelo.getVal(y[a]) > 0.5), None)
        ordenes = [int(modelo.getVal(z[o]) + 0.5) for o in range(O)]
        unidades = sum(units_o[o] for o in range(O) if ordenes[o])

        return {'pasillo': pasillo_seleccionado, 'ordenes': ordenes, 'unidades': unidades}


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

            print(f"⌛ Iteración con {len(self.columnas.get(k, []))} columnas")

            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_pasillos = self.construir_modelo_maestro(k, tiempo_restante_total)
            if maestro is None:
                print("No se pudo construir el modelo maestro a tiempo")
                return None

            maestro_relajado = Model(sourceModel=maestro)
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()
            for var in maestro_relajado.getVars():
                maestro_relajado.chgVarType(var, "CONTINUOUS")
            maestro_relajado.optimize()

            if maestro_relajado.getStatus() != "optimal":
                print("⚠️ No se encontró solución. Estado del modelo:", maestro_relajado.getStatus())
                break

            dual_map = {cons.name: maestro_relajado.getDualSolVal(cons) for cons in maestro_relajado.getConss()}

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False


            valor_objetivo_primal = maestro_relajado.getObjVal()

            mejor_sol = construir_mejor_solucion(maestro_relajado, self.columnas.get(k, []), valor_objetivo_primal, self.cant_var_inicio)
            

            tiempo_restante_total = umbral - (time.time() - tiempo_ini)
            nueva_col = self.resolver_subproblema(self.W, self.S, dual_map, self.UB, k, tiempo_restante_total)


            if nueva_col is None :
                print("No se generó columna nueva → Fin del bucle.")
                break

            print("Nueva columna encontrada:", nueva_col)
            self.columnas.setdefault(k, []).append(nueva_col)

        return mejor_sol




    def Opt_PasillosFijos(self, umbral):
        tiempo_ini = time.time()
        k = len(self.pasillos_fijos)
        solucion_vacia = {"valor_objetivo": 0,"pasillos_seleccionados": set(),"ordenes_seleccionadas": set(),"restricciones": 0,"variables": 0,"variables_final": 0,"cota_dual": 0}
        
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

        if status in ["optimal", "feasible"] and modelo.getNSols() > 0:
            obj_val = modelo.getObjVal()
            pasillos_seleccionados = set()
            ordenes_seleccionadas = set()

            for idx, x in enumerate(x_vars):
                val = x.getLPSol()
                if val and val > 1e-5:
                    pasillos_seleccionados.add(self.columnas[k][idx]['pasillo'])
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

        lista_k, lista_umbrales = self.Rankear(umbral)

        for k, tiempo_k in zip(lista_k, lista_umbrales):
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_ini
            tiempo_restante_total = umbral - 4 - tiempo_transcurrido

            if tiempo_restante_total <= 0:
                break

            tiempo_k_deseado = tiempo_k
            tiempo_k = min(tiempo_k_deseado, tiempo_restante_total)

            print(f"Evaluando k={k} con tiempo asignado {tiempo_k:.2f} segundos")

            sol = self.Opt_cantidadPasillosFija(k, tiempo_k)

            if sol:
                sol_obj = sol.get("valor_objetivo", -float('inf'))
                best_obj = best_sol.get("valor_objetivo", -float('inf')) if best_sol else -float('inf')
                if sol_obj > best_obj:
                    best_sol = sol

        if best_sol:
            tiempo_usado = time.time() - tiempo_ini
            tiempo_final = max(1.0, umbral - tiempo_usado)

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


    def Rankear(self, umbral):
        capacidades = [sum(self.S[a]) for a in range(self.A)]
        pasillos_ordenados = sorted(range(1, self.A + 1), key=lambda a: capacidades[a-1], reverse=True)
        lista_k = pasillos_ordenados
        tiempo_por_k = umbral 
        lista_umbrales = [tiempo_por_k] * len(lista_k)

        return lista_k, lista_umbrales


