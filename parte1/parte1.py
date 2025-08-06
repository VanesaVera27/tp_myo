from pyscipopt import Model, quicksum
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cargar_input import leer_input
from guardar_output import guardar_resultado_estandar


def crear_modelo(W, S, LB, UB, max_pasillos):
    n_ordenes = len(W)
    n_elementos = len(W[0])
    n_pasillos = len(S)

    modelo = Model("Desafio_Parte1")
    modelo.setParam('display/verblevel', 0)

    x = {o: modelo.addVar(vtype="B", name=f"x_{o}") for o in range(n_ordenes)}
    y = {a: modelo.addVar(vtype="B", name=f"y_{a}") for a in range(n_pasillos)}

    total_recolectado = quicksum(W[o][i] * x[o] for o in range(n_ordenes) for i in range(n_elementos))
    modelo.setObjective(total_recolectado / max_pasillos, "maximize")

    modelo.addCons(total_recolectado >= LB, "LB")
    modelo.addCons(total_recolectado <= UB, "UB")

    for i in range(n_elementos):
        modelo.addCons(
            quicksum(W[o][i] * x[o] for o in range(n_ordenes)) <=
            quicksum(S[a][i] * y[a] for a in range(n_pasillos)),
            name=f"disp_elem_{i}"
        )

    modelo.addCons(quicksum(y[a] for a in range(n_pasillos)) == max_pasillos, "cant_pasillos")

    return modelo, x, y


def obtener_solucion(modelo, x, y, n_ordenes, n_pasillos):
    x_sol = [int(modelo.getVal(x[o])) for o in range(n_ordenes)]
    y_sol = [int(modelo.getVal(y[a])) for a in range(n_pasillos)]
    return x_sol, y_sol


def imprimir_solucion(modelo, x_sol, y_sol):
    print("Órdenes seleccionadas (O'):", [o for o, val in enumerate(x_sol) if val == 1])
    print("Pasillos seleccionados (A'):", [a for a, val in enumerate(y_sol) if val == 1])
    print("Valor objetivo (items por pasillo):", modelo.getObjVal())


def es_factible(x_sol, y_sol, W, S, LB, UB, max_pasillos):
    n_ordenes = len(W)
    n_elementos = len(W[0])
    n_pasillos = len(S)

    total_unidades = sum(sum(W[o][i] for i in range(n_elementos)) for o in range(n_ordenes) if x_sol[o] == 1)

    if total_unidades < LB:
        print("Falla en restricción LB")
        return False
    if total_unidades > UB:
        print("Falla en restricción UB")
        return False

    for i in range(n_elementos):
        pedida = sum(W[o][i] * x_sol[o] for o in range(n_ordenes))
        disponible = sum(S[a][i] * y_sol[a] for a in range(n_pasillos))
        if pedida > disponible:
            print(f"Falla en restricción disponibilidad para el elemento {i}")
            return False

    if sum(y_sol) != max_pasillos:
        print("Falla en restricción de cantidad de pasillos seleccionados")
        return False

    return True


if __name__ == "__main__":
    archivo_input = "datos_de_entrada/A/instance_0001.txt"
    W, S, LB, UB = leer_input(archivo_input)
    max_pasillos = 2

    n_ordenes = len(W)
    n_pasillos = len(S)

    modelo, x, y = crear_modelo(W, S, LB, UB, max_pasillos)
    modelo.optimize()
    status = modelo.getStatus()
    print("Estado de solución:", status)

    if status in ["optimal", "optimal_inaccurate"]:
        x_sol, y_sol = obtener_solucion(modelo, x, y, n_ordenes, n_pasillos)
        imprimir_solucion(modelo, x_sol, y_sol)

        if es_factible(x_sol, y_sol, W, S, LB, UB, max_pasillos):
            print("La solución es factible.")
        else:
            print("La solución NO es factible.")
    else:
        print("No se encontró solución óptima.")

    guardar_resultado_estandar(
        archivo_input=archivo_input,
        parte="parte1",
        status=status,
        modelo=modelo,
        x_vars=x,
        y_vars=y,
        n_ordenes=n_ordenes,
        n_pasillos=n_pasillos
    )


