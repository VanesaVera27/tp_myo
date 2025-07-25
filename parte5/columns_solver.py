import time
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

def construir_mejor_solucion(modelo_relajado, columnas_k, obj_val, cant_var_inicio):
    pasillos_seleccionados = set()
    ordenes_seleccionadas = set()

    for idx, x in enumerate(modelo_relajado.getVars()):
        if x.getLPSol() and x.getLPSol() > 1e-5:
            pasillos_seleccionados.add(columnas_k[idx]['pasillo'])
            for o, val in enumerate(columnas_k[idx]['ordenes']):
                if val:
                    ordenes_seleccionadas.add(o)

    mejor_sol = {
        "valor_objetivo": obj_val / len(pasillos_seleccionados) if pasillos_seleccionados else 0,
        "pasillos_seleccionados": pasillos_seleccionados,
        "ordenes_seleccionadas": ordenes_seleccionadas,
        "variables": cant_var_inicio,
        "variables_final": modelo_relajado.getNVars(),
        "cota_dual": modelo_relajado.getDualbound()
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

        for a in range(self.A):  # Recorremos cada pasillo
            if umbral and (time.time() - tiempo_ini) > umbral:
                print("⏱️ Tiempo agotado durante inicialización de columnas")
                break

            cap_restante = list(self.S[a])
            sel = [0] * self.O
            total_unidades = 0

            # Agregar órdenes greedy mientras entren (maximales)
            for o in range(self.O):
                if all(self.W[o][i] <= cap_restante[i] for i in range(self.I)) and \
                (total_unidades + unidades_o[o] <= self.UB):
                    sel[o] = 1
                    total_unidades += unidades_o[o]
                    for i in range(self.I):
                        cap_restante[i] -= self.W[o][i]

            # Siempre agregar una columna por pasillo (aunque sea vacía)
            self.columnas[k].append({
                'pasillo': a,
                'ordenes': sel,  # Puede tener todo 0 si no había órdenes factibles
                'unidades': total_unidades
            })

        print(f"✅ {len(self.columnas[k])} columnas iniciales creadas (una por pasillo) para k = {k}")

    def construir_modelo_maestro(self, k, umbral):
        modelo = Model(f"RMP_k_{k}")
        modelo.setParam('display/verblevel', 0)
        x_vars = []
        
        # variable x_j binaria
        for idx, col in enumerate(self.columnas[k]):
            x = modelo.addVar(vtype="B", name=f"x_{col['pasillo']}_{idx}")
            x_vars.append(x)

        # Restricción de cardinalidad
        restr_card_k = modelo.addCons(quicksum(x_vars) == k, name="card_k")

        # Restricciones por orden
        restr_ordenes = {}
        for o in range(self.O):
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

        # restr_lb = modelo.addCons(
        #     quicksum(x_vars[j] * self.columnas[k][j]['unidades'] for j in range(len(x_vars))) >= self.UB,
        #     name="restr_total_lb"
        # )

        # Restricciones de cobertura por ítem
        restr_cov = {}
        for i in range(self.I):
            cons = modelo.addCons(
                quicksum(
                    x_vars[j] * sum(self.W[o][i] for o in range(self.O) if self.columnas[k][j]['ordenes'][o])
                    for j in range(len(x_vars))
                ) <= quicksum(
                    x_vars[j] * self.S[self.columnas[k][j]['pasillo']][i]
                    for j in range(len(x_vars))
                ),
                name=f"cobertura_item_{i}"
            )
            restr_cov[i] = cons

        for a in range(self.A):
            modelo.addCons(
                quicksum(
                    x_vars[j] for j in range(len(x_vars)) if self.columnas[k][j]['pasillo'] == a
                ) <= 1,
                name=f"pasillo_{a}"
            )

        # Función objetivo: maximizar la suma total de unidades recolectadas (por orden y ítem)
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

        return modelo, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov

    def resolver_subproblema(self, W, S, pi, UB, k, umbral=None):
        tiempo_ini = time.time()

        O = len(W)
        I = len(W[0])
        A = len(S)

        units_o = [sum(W[o]) for o in range(O)]

        # Duales obtenidos del modelo maestro, en orden:
        # pi[0]       = γ_k (dual de cardinalidad fija)
        # pi[1:1+O]   = α_o (duales de cobertura por orden)
        # pi[1+O]     = λ (dual de límite superior unidades totales)
        # pi[2+O:2+O+I] = π_i (duales de cobertura por ítem)
        pi_card_k = pi[0]
        pi_ordenes = pi[1:1 + O]
        pi_ub = pi[1 + O]
        pi_cov = pi[2 + O: 2 + O + I]
        pi_pasillos = pi[2 + O + I: 2 + O + I + A]

        modelo = Model("Subproblema_unico")
        modelo.setParam("display/verblevel", 0)

        # Variables:
        # y_a: binaria, indica si se selecciona el pasillo a
        y = {a: modelo.addVar(vtype="B", name=f"y_{a}") for a in range(A)}

        # z_o: binaria, indica si se selecciona la orden o en la columna
        z = {o: modelo.addVar(vtype="B", name=f"z_{o}") for o in range(O)}

        # Restricción: seleccionar exactamente un pasillo
        modelo.addCons(quicksum(y[a] for a in range(A)) == 1, name="unico_pasillo")

        # Restricción: capacidad por ítem condicionada al pasillo seleccionado
        for i in range(I):
            modelo.addCons(
                quicksum(W[o][i] * z[o] for o in range(O)) <=
                quicksum(S[a][i] * y[a] for a in range(A)),
                name=f"capacidad_item_{i}"
            )

        # Restricción: límite superior de unidades totales
        modelo.addCons(quicksum(units_o[o] * z[o] for o in range(O)) <= UB, name="limite_unidades")

        # Al menos una orden debe ser seleccionada
        modelo.addCons(quicksum(z[o] for o in range(O)) >= 1, name="min_una_orden")

        # Evitar columnas repetidas (de todas las columnas previas)
        for k_col in self.columnas.get(k, []):
            mismos = [o for o in range(O) if k_col['ordenes'][o] == 1]
            if mismos:
                modelo.addCons(
                    quicksum(z[o] for o in mismos) <= len(mismos) - 1,
                    name=f"evitar_repetida_{k_col['pasillo']}"
                )

        # Calcular costo reducido por orden, independiente del pasillo porque la cobertura está condicionada a y
        price_o = []
        for o in range(O):
            rc_part = units_o[o]                                    # uo
            rc_part -= sum(W[o][i] * pi_cov[i] for i in range(I))  # - ∑ π_i * W[o][i]
            rc_part -= pi_ub * units_o[o]                           # - λ * uo
            rc_part -= pi_ordenes[o]                                # - α_o
            price_o.append(rc_part)

        # Función objetivo: maximizar costo reducido menos duales
        modelo.setObjective(
            quicksum(price_o[o] * z[o] for o in range(O)) -
            quicksum((y[a] for a in range(A))) - pi_card_k,
            sense="maximize"
        )


        # Optimizar
        modelo.optimize()

        if modelo.getStatus() != "optimal":
            print("⚠️ Subproblema no óptimo.")
            return None

        reduced_cost = modelo.getObjVal()
        print(f"Costo reducido subproblema: {reduced_cost}")

        if reduced_cost <= 1e-6:
            return None

        # Obtener el pasillo seleccionado
        pasillo_seleccionado = None
        for a in range(A):
            if modelo.getVal(y[a]) > 0.5:
                pasillo_seleccionado = a
                break

        # Obtener las órdenes seleccionadas
        ordenes = [int(modelo.getVal(z[o]) + 0.5) for o in range(O)]
        unidades = sum(units_o[o] for o in range(O) if ordenes[o])

        columna = {'pasillo': pasillo_seleccionado, 'ordenes': ordenes, 'unidades': unidades}

        if columna in self.columnas.get(k, []):
            print("⚠️ Columna repetida generada:", columna)
            return None

        self.columnas.setdefault(k, []).append(columna)
        return columna


    def agregar_columna(self, maestro, nueva_col, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov, k):
        idx = len(self.columnas[k]) - 1  # índice correcto de la nueva columna (ya agregada)
        x = maestro.addVar(vtype="B", name=f"x_{nueva_col['pasillo']}_{idx}", obj=nueva_col['unidades'])
        x_vars.append(x)

        # Agregar coeficiente en restricción cardinalidad
        maestro.addConsCoeff(restr_card_k, x, 1)

        # Agregar coeficientes en restricción de órdenes
        for o in range(self.O):
            maestro.addConsCoeff(restr_ordenes[o], x, nueva_col['ordenes'][o])

        # Agregar coeficiente en restricción límite superior unidades
        maestro.addConsCoeff(restr_ub, x, nueva_col['unidades'])

        # Agregar coeficientes en restricciones de cobertura por ítem
        for i in range(self.I):
            contribucion_i = sum(self.W[o][i] * nueva_col['ordenes'][o] for o in range(self.O))
            if contribucion_i > 0:
                maestro.addConsCoeff(restr_cov[i], x, contribucion_i)

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
            maestro, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov = self.construir_modelo_maestro(k, tiempo_restante_total)

            # Crear una copia para relajación y obtención de duales
            maestro_relajado = Model(sourceModel=maestro)

            # Desactivar ciertas heurísticas y preprocesos para obtener duales confiables
            maestro_relajado.setPresolve(SCIP_PARAMSETTING.OFF)
            maestro_relajado.setHeuristics(SCIP_PARAMSETTING.OFF)
            maestro_relajado.disablePropagation()

            # Relajar modelo maestro (variables binarias a continuas)
            maestro_relajado.relax()
            maestro_relajado.optimize()

            if primera_iteracion:
                self.cant_var_inicio = maestro_relajado.getNVars()
                primera_iteracion = False

            # Verificar que la solución es óptima para obtener duales
            if maestro_relajado.getStatus() != "optimal":
                print(f"⚠️ No se encontró solución óptima. Estado: {maestro_relajado.getStatus()}")
                break

            # Obtener valores duales de las restricciones para usar en el subproblema
            pi = [maestro_relajado.getDualsolLinear(c) for c in maestro_relajado.getConss()]

            # Guardar valor objetivo actual
            valor_objetivo = maestro_relajado.getObjVal()
            # print("Valor objetivo:", valor_objetivo)

            # Construir la mejor solución factible a partir del modelo relajado
            mejor_sol = construir_mejor_solucion(maestro_relajado, self.columnas.get(k, []), valor_objetivo, self.cant_var_inicio)

            # print("❤️ Cantidad de columnas antes de agregar:", len(self.columnas.get(k, [])))

            # Resolver el subproblema para encontrar una columna con costo reducido negativo
            nueva_col = self.resolver_subproblema(self.W, self.S, pi, self.UB, k, tiempo_restante_total)

            if nueva_col is None:
                print("No se generó una columna mejoradora o era repetida → Fin del bucle.")
                break

            print("Nueva columna encontrada:", nueva_col)
            print("➕ Agregando columna nueva al modelo maestro.")

            # Agregar la nueva columna al modelo maestro
            self.agregar_columna(maestro, nueva_col, x_vars, restr_card_k, restr_ordenes, restr_ub, restr_cov, k)

            # print("💙 Cantidad de columnas después de agregar:", len(self.columnas.get(k, [])))

        return mejor_sol


    def Opt_PasillosFijos(self, umbral):
        tiempo_ini = time.time()
        k = len(self.pasillos_fijos)
        
        # Calcular tiempo restante correctamente (usamos tiempo_ini para referencia)
        tiempo_restante_final = umbral - (time.time() - tiempo_ini)
        if tiempo_restante_final <= 0:
            print("⏳ No queda tiempo para Opt_PasillosFijos")
            return {
                "valor_objetivo": 0,
                "pasillos_seleccionados": set(),
                "ordenes_seleccionadas": set(),
                "restricciones": 0,
                "variables": 0,
                "variables_final": 0,
                "cota_dual": 0
            }

        # Validar que haya columnas para k
        if k not in self.columnas or not self.columnas[k]:
            print(f"❌ No hay columnas generadas para k = {k}")
            return {
                "valor_objetivo": 0,
                "pasillos_seleccionados": set(),
                "ordenes_seleccionadas": set(),
                "restricciones": 0,
                "variables": 0,
                "variables_final": 0,
                "cota_dual": 0
            }

        # Construir el modelo maestro con las columnas actuales
        modelo, x_vars, _, _, _, _ = self.construir_modelo_maestro(k, tiempo_restante_final)

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
                "variables": self.cant_var_inicio if hasattr(self, "cant_var_inicio") else 0,
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
                "variables": self.cant_var_inicio if hasattr(self, "cant_var_inicio") else 0,
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
        # Calcular la capacidad total por pasillo
        capacidades = [sum(self.S[a]) for a in range(self.A)]
        # Ordenar los pasillos por capacidad de mayor a menor
        pasillos_ordenados = sorted(range(self.A), key=lambda a: capacidades[a], reverse=True)
        lista_k = pasillos_ordenados

        # Tiempo asignado por k, equitativamente
        tiempo_por_k = umbral 
        lista_umbrales = [tiempo_por_k] * len(lista_k)

        return lista_k, lista_umbrales


# OTRA ALTERNATIVA DE PASILLOS FIJOS, COMO EN PARTE 2

    # def Opt_PasillosFijos(self, umbral):
    #     if not self.pasillos_fijos:
    #         print("No hay pasillos fijos definidos para Opt_PasillosFijos")
    #         return None

    #     start = time.time()
    #     model = Model("Opt_PasillosFijos")
    #     model.setParam('display/verblevel', 0)  # silenciar logs
    #     model.setParam('limits/nodes', 5000)  # limitar nodos
    #     model.setParam("limits/time", umbral)

    #     x = {}
    #     for o in range(self.O):
    #         x[o] = model.addVar(vtype="B", name=f"x_{o}")

    #     model.setObjective(
    #         quicksum(self.W[o][i] * x[o] for o in range(self.O) for i in range(self.I)),
    #         "maximize"
    #     )

    #     total = quicksum(self.W[o][i] * x[o] for o in range(self.O) for i in range(self.I))
    #     model.addCons(total >= self.LB)
    #     model.addCons(total <= self.UB)

    #     for i in range(self.I):
    #         disponibilidad_total = sum(self.S[a][i] for a in self.pasillos_fijos)
    #         model.addCons(quicksum(self.W[o][i] * x[o] for o in range(self.O)) <= disponibilidad_total)

    #     cantidad_restricciones = model.getNConss()
    #     model.optimize()

    #     if model.getStatus() != "optimal":
    #         print(f"Modelo no óptimo: {model.getStatus()}")
    #         return None

    #     ordenes_seleccionadas = [o for o in range(self.O) if model.getVal(x[o]) > 0.5]
    #     valor_objetivo = model.getObjVal()

    #     return {
    #         "valor_objetivo": valor_objetivo,
    #         "ordenes_seleccionadas": ordenes_seleccionadas,
    #         "pasillos_seleccionados": self.pasillos_fijos,
    #         "restricciones": cantidad_restricciones,
    #         "variables": len(x),
    #         "variables_final": len(x),
    #         "cota_dual": model.getDualbound()
    #     }

