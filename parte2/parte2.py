from pyscipopt import Model, quicksum
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cargar_input import leer_input
from guardar_output import guardar_resultado_simple


def crear_modelo(W, S, LB, UB, pasillos):
    n_ordenes = len(W)
    n_elementos = len(W[0])

    modelo = Model("Desafio_Parte2")
    modelo.setParam('display/verblevel', 0)

    x = {o: modelo.addVar(vtype="B", name=f"x_{o}") for o in range(n_ordenes)}

    total_recolectado = quicksum(W[o][i] * x[o] for o in range(n_ordenes) for i in range(n_elementos))
    modelo.setObjective(total_recolectado, "maximize")

    modelo.addCons(total_recolectado >= LB, "LB")
    modelo.addCons(total_recolectado <= UB, "UB")

    for i in range(n_elementos):
        disponibilidad_total = sum(S[a][i] for a in pasillos)
        modelo.addCons(
            quicksum(W[o][i] * x[o] for o in range(n_ordenes)) <= disponibilidad_total,
            name=f"disp_elem_{i}"
        )

    return modelo, x


def obtener_solucion_x(modelo, x_vars, n_ordenes):
    return [int(modelo.getVal(x_vars[o])) for o in range(n_ordenes)]


def imprimir_solucion(modelo, x_sol):
    ordenes_seleccionadas = [o for o, val in enumerate(x_sol) if val == 1]
    print("Órdenes seleccionadas (O'):", ordenes_seleccionadas)
    print("Unidades recolectadas:", modelo.getObjVal())


if __name__ == "__main__":
    archivo_input = "datos_de_entrada/A/instance_0020.txt"
    W, S, LB, UB = leer_input(archivo_input)

    pasillos = [2, 4, 0, 4]  # pasillos seleccionados manualmente
    n_ordenes = len(W)

    modelo, x_vars = crear_modelo(W, S, LB, UB, pasillos)
    modelo.optimize()
    status = modelo.getStatus()
    print("Estado de solución:", status)

    if status in ["optimal", "optimal_inaccurate"]:
        x_sol = obtener_solucion_x(modelo, x_vars, n_ordenes)
        imprimir_solucion(modelo, x_sol)
    else:
        print("No se encontró solución óptima.")

    # Guardar resultados
    guardar_resultado_simple(
        archivo_input=archivo_input,
        parte="parte2",
        status=status,
        modelo=modelo,
        x_vars=x_vars,
        n_ordenes=n_ordenes,
        pasillos_usados=pasillos
    )



