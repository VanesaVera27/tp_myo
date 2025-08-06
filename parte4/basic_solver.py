from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
import time


class Basic:
    def __init__(self, W, S, LB, UB):
        self.W = W
        self.S = S
        self.LB = LB
        self.UB = UB
        self.n_ordenes = len(W)
        self.n_elementos = len(W[0])
        self.n_pasillos = len(S)
        self.mejores_pasillos = set()
        self.mejor_solucion = None
        self.ultima_cota_dual = -float("inf")
        self.mejor_cota_dual = -float("inf")
        self.modelo_maestro, self.restriccion_k = self.construir_maestro()

    def construir_maestro(self):
        maestro = Model("Modelo maestro") 
        maestro.setPresolve(SCIP_PARAMSETTING.OFF)
        maestro.setParam("display/verblevel", 0)

        x = {a: maestro.addVar(vtype="B", name=f"x_{a}") for a in range(self.n_pasillos)}
        y = {o: maestro.addVar(vtype="B", name=f"y_{o}") for o in range(self.n_ordenes)}

        for i in range(self.n_elementos):
            maestro.addCons( quicksum(self.W[o][i] * y[o] for o in y) <= quicksum(self.S[a][i] * x[a] for a in x),
                name=f"cap_{i}"
            )

        restriccion_k = maestro.addCons(quicksum(x.values()) == 1, name="cons_k")
        total = quicksum(self.W[o][i] * y[o] for o in y for i in range(self.n_elementos))

        maestro.addCons(total >= self.LB, name="LB")
        maestro.addCons(total <= self.UB, name="UB")
        maestro.setObjective(total, "maximize")

        return maestro, restriccion_k

    def modelo_para_k(self, K):
        try:
            modelo_copia = self.modelo_maestro.copyOrig()
            restriccion_k = modelo_copia.getCons("cons_k")
        except AttributeError:
            modelo_copia, restriccion_k = self.construir_maestro()

        modelo_copia.chgRhs(restriccion_k, K)
        return modelo_copia

    def Rankear(self):
        return list(range(1, self.n_pasillos + 1))

    def Opt_cantidadPasillosFija(self, k, umbral):
        modelo = self.modelo_para_k(k)

        if self.mejor_solucion and len(self.mejor_solucion["pasillos_seleccionados"]) == k:
            sol = modelo.createSol()
            
            pasillos = self.mejor_solucion["pasillos_seleccionados"]
            ordenes = self.mejor_solucion["ordenes_seleccionadas"]

            for var in modelo.getVars():
                nombre, indice = var.name.split("_")
                indice = int(indice)
                if nombre == "x":
                    valor = 1.0 if indice in pasillos else 0.0
                elif nombre == "y":
                    valor = 1.0 if indice in ordenes else 0.0
                else:
                    continue  
                modelo.setSolVal(sol, var, valor)
            modelo.addSol(sol, False)

        if umbral:
            modelo.setParam("limits/time", umbral)

        modelo.optimize()
        self.ultima_cota_dual = modelo.getDualbound()
        if self.ultima_cota_dual > self.mejor_cota_dual:
            self.mejor_cota_dual = self.ultima_cota_dual

        if modelo.getStatus() == "optimal" or modelo.getStatus() == "feasible":
            pasillos = {int(v.name.split("_")[1]) for v in modelo.getVars()if v.name.startswith("x_") and modelo.getVal(v) > 0.5}
            ordenes = {int(v.name.split("_")[1]) for v in modelo.getVars()if v.name.startswith("y_") and modelo.getVal(v) > 0.5}
            total = sum(sum(self.W[o][i] for i in range(self.n_elementos)) for o in ordenes)
            sol = { "valor_objetivo": total / len(pasillos) if pasillos else 0, "ordenes_seleccionadas": ordenes, "pasillos_seleccionados": pasillos}
            self.mejor_solucion = sol
            return sol
        else:
            return None

    def Opt_PasillosFijos(self, umbral):
        k = len(self.mejores_pasillos)
        modelo = self.modelo_para_k(k)

        for var in modelo.getVars():
            if var.name.startswith("x_"):
                idx = int(var.name.split("_")[1])
                if idx in self.mejores_pasillos:
                    modelo.chgVarLb(var, 1.0)
                    modelo.chgVarUb(var, 1.0)
                else:
                    modelo.chgVarUb(var, 0.0)


        if umbral:
            modelo.setParam("limits/time", umbral)

        modelo.optimize()

        if modelo.getStatus() == "optimal" or modelo.getStatus() == "feasible":
            pasillos = {int(v.name.split("_")[1]) for v in modelo.getVars() if v.name.startswith("x_") and modelo.getVal(v) > 0.5}
            ordenes = {int(v.name.split("_")[1]) for v in modelo.getVars() if v.name.startswith("y_") and modelo.getVal(v) > 0.5}
            total = sum(sum(self.W[o][i] for i in range(self.n_elementos)) for o in ordenes)
            sol = { "valor_objetivo": total / len(pasillos) if pasillos else 0, "ordenes_seleccionadas": ordenes, "pasillos_seleccionados": pasillos}
            self.mejor_solucion = sol
            return sol
        else:
            return None
        
    # def Opt_PasillosFijos(self, umbral):
    #     start = time.time()
    #     if not self.mejores_pasillos:
    #         raise RuntimeError("Primero ejecuta Opt_ExplorarCantidadPasillos o fija los pasillos.")

    #     modelo = Model("Opt_PasillosFijos")
    #     modelo.setPresolve(SCIP_PARAMSETTING.OFF)
    #     modelo.setParam('display/verblevel', 0)
        
    #     x = {o: modelo.addVar(vtype="B", name=f"x_{o}") for o in range(self.n_ordenes)}

    #     total = quicksum(x[o] * sum(self.W[o][i] for i in range(self.n_elementos)) for o in range(self.n_ordenes))
    #     modelo.setObjective(total, "maximize")

    #     modelo.addCons(total >= self.LB, name="LB")
    #     modelo.addCons(total <= self.UB, name="UB")

    #     for i in range(self.n_elementos):
    #         capacidad = sum(self.S[a][i] for a in self.mejores_pasillos)
    #         modelo.addCons(quicksum(self.W[o][i] * x[o] for o in range(self.n_ordenes)) <= capacidad, name=f"cap_{i}")

    #     if time.time() - start > umbral:
    #         return None

    #     modelo.optimize()
    #     if modelo.getStatus() != "optimal":
    #         return None

    #     ordenes = [o for o in range(self.n_ordenes) if modelo.getVal(x[o]) > 0.5]
    #     valor = modelo.getObjVal()

    #     return {
    #         "valor_objetivo": valor/len(self.mejores_pasillos),
    #         "ordenes_seleccionadas": ordenes,
    #         "pasillos_seleccionados": self.mejores_pasillos
    #     }

    def Opt_ExplorarCantidadPasillos(self, umbral_total):
        start = time.time()
        mejor_sol = None
        mejor_valor = -float("inf")
        k_list = self.Rankear()

        for k in k_list:
            tiempo_restante = umbral_total - (time.time() - start)
            if tiempo_restante <= 0:
                break

            solucion = self.Opt_cantidadPasillosFija(k, tiempo_restante)
            if solucion and solucion["valor_objetivo"] > mejor_valor:
                mejor_valor = solucion["valor_objetivo"]
                mejor_sol = solucion

        if mejor_sol:
            self.mejores_pasillos = mejor_sol["pasillos_seleccionados"]
            tiempo_restante = umbral_total - (time.time() - start)
            if tiempo_restante > 0:
                solucion_para_k_fijo = self.Opt_PasillosFijos(tiempo_restante)
                if solucion_para_k_fijo and solucion_para_k_fijo["valor_objetivo"] > mejor_valor:
                    mejor_sol = solucion_para_k_fijo

        self.mejor_solucion = mejor_sol
        return mejor_sol