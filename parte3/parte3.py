from pyscipopt import Model, quicksum
from itertools import combinations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cargar_input import leer_input
from guardar_output import guardar_resultado_columnas

class ModeloColumnas:
    def __init__(self, archivo_input, leer_input_func, max_r=2, max_patrones_por_pasillo=10, max_pasillos=2):
        self.archivo_input = archivo_input
        self.W, self.S, self.LB, self.UB = leer_input_func(archivo_input)

        self.O = len(self.W)           # Cantidad de órdenes
        self.I = len(self.W[0])        # Cantidad de ítems
        self.A = len(self.S)           # Cantidad de pasillos
        self.k = max_pasillos

        self.max_r = max_r
        self.max_patrones = max_patrones_por_pasillo
        self.columnas = []             # Lista de columnas/patrones
        self.modelo = None
        self.x_vars = []

        self.generar_columnas_iniciales()

    def es_patron_factible(self, ordenes, pasillo):
        for i in range(self.I):
            demanda_total = sum(self.W[o][i] for o in ordenes)
            if demanda_total > self.S[pasillo][i]:
                return False
        return True

    def generar_columnas_iniciales(self):
        for a in range(self.A):
            cuenta = 0
            for r in range(1, self.max_r + 1):
                for subset in combinations(range(self.O), r):
                    if self.es_patron_factible(subset, a):
                        ordenes_vector = [0] * self.O
                        for o in subset:
                            ordenes_vector[o] = 1
                        unidades = sum(self.W[o][i] for o in subset for i in range(self.I))

                        columna = {
                            'ordenes': ordenes_vector,
                            'pasillo': a,
                            'unidades': unidades
                        }
                        self.columnas.append(columna)
                        cuenta += 1
                        if cuenta >= self.max_patrones:
                            break
                if cuenta >= self.max_patrones:
                    break

    def ConstruirModelo(self):
        self.modelo = Model("Desafio_Parte3")
        self.modelo.setParam('display/verblevel', 0)
        self.x_vars = []

        for idx, col in enumerate(self.columnas):
            x = self.modelo.addVar(vtype="B", name=f"x_{col['pasillo']}_{idx}")
            self.x_vars.append(x)

        # Cardinalidad (cantidad de pasillos)
        self.modelo.addCons(quicksum(self.x_vars) == self.k, name="card_k")

        # Restricción por orden: una vez como máximo
        for o in range(self.O):
            self.modelo.addCons(
                quicksum(self.x_vars[j] * self.columnas[j]['ordenes'][o] for j in range(len(self.x_vars))) <= 1,
                name=f"orden_{o}"
            )

        # Restricción total de unidades
        self.modelo.addCons(
            quicksum(self.x_vars[j] * self.columnas[j]['unidades'] for j in range(len(self.x_vars))) <= self.UB,
            name="restr_total_ub"
        )

        # Restricciones de cobertura por ítem
        for i in range(self.I):
            self.modelo.addCons(
                quicksum(
                    self.x_vars[j] * sum(self.W[o][i] for o in range(self.O) if self.columnas[j]['ordenes'][o])
                    for j in range(len(self.x_vars))
                ) <= quicksum(
                    self.x_vars[j] * self.S[self.columnas[j]['pasillo']][i]
                    for j in range(len(self.x_vars))
                ),
                name=f"cobertura_item_{i}"
            )

        # Un solo patrón por pasillo
        for a in range(self.A):
            self.modelo.addCons(
                quicksum(
                    self.x_vars[j] for j in range(len(self.x_vars)) if self.columnas[j]['pasillo'] == a
                ) <= 1,
                name=f"pasillo_{a}"
            )

        # Objetivo: maximizar unidades cubiertas
        self.modelo.setObjective(
            quicksum(
                self.x_vars[j] * sum(
                    self.W[o][i]
                    for o in range(self.O) if self.columnas[j]['ordenes'][o]
                    for i in range(self.I)
                )
                for j in range(len(self.x_vars))
            ),
            sense="maximize"
        )

    def AgregarColumna(self, columna):
        self.modelo.freeTransform()  # Libera la fase de solución

        idx = len(self.columnas)
        self.columnas.append(columna)

        x = self.modelo.addVar(vtype="B", name=f"x_{columna['pasillo']}_{idx}")
        self.x_vars.append(x)

        # Restricciones
        for o in range(self.O):
            if columna['ordenes'][o] == 1:
                self.modelo.addCons(x <= 1, name=f"orden_{o}_x_{idx}")

        self.modelo.addCons(x * columna['unidades'] <= self.UB, name=f"ub_x_{idx}")

        for i in range(self.I):
            carga_orden = sum(self.W[o][i] for o in range(self.O) if columna['ordenes'][o])
            capacidad = self.S[columna['pasillo']][i]
            self.modelo.addCons(
                x * carga_orden <= x * capacidad,
                name=f"cov_x_{idx}_item_{i}"
            )

        self.modelo.addCons(x <= 1, name=f"pasillo_{columna['pasillo']}_x_{idx}")

    def resolver(self):
        self.modelo.optimize()
        if self.modelo.getStatus() == "optimal":

            ordenes_seleccionadas = set()
            pasillos_seleccionados = set()
            total_unidades = 0

            for i, x in enumerate(self.x_vars):
                if self.modelo.getVal(x) > 0.5:
                    col = self.columnas[i]
                    pasillos_seleccionados.add(col['pasillo'])

                    # Sumar unidades recolectadas en esta columna
                    total_unidades += col['unidades']

                    # Agregar órdenes activas (donde hay un 1 en el vector 'ordenes')
                    ordenes_seleccionadas.update(
                        o for o, val in enumerate(col['ordenes']) if val == 1
                    )

            if pasillos_seleccionados:
                valor_objetivo = total_unidades / len(pasillos_seleccionados)
                print(f"Mejor valor objetivo: {valor_objetivo}")
                print(f"Órdenes seleccionadas: {ordenes_seleccionadas}")
                print(f"Pasillos seleccionados: {pasillos_seleccionados}")
                return valor_objetivo, ordenes_seleccionadas, pasillos_seleccionados
            else:
                print("No se seleccionaron pasillos.")

        else:
            print("No se encontró solución óptima.")


if __name__ == "__main__":
    archivo = "datos_de_entrada/A/instance_0020.txt"
    modelo = ModeloColumnas(archivo_input=archivo, leer_input_func=leer_input, max_r=2, max_patrones_por_pasillo=10, max_pasillos=2)
    modelo.ConstruirModelo() 
    print("Resuelvo el modelo inicialmente ===========")
    _,_,_ = modelo.resolver()
    print("\n===========================================")
    print("Cantidad de columnas antes de agregar:", len(modelo.columnas))

    # Agregar columna factible manualmente
    nueva_col = {
        'ordenes': [1 if o == 0 else 0 for o in range(modelo.O)],
        'pasillo': 0,
        'unidades': sum(modelo.W[0])
    }

    print("Agrego una columna nueva ✅")
    modelo.AgregarColumna(nueva_col)
    print("Cantidad de columnas despues de agregar:", len(modelo.columnas))
    print("===========================================")

    print("\nResuelvo el modelo nuevamente =============")
    valor_objetivo, ordenes, pasillos = modelo.resolver()

    guardar_resultado_columnas(
        archivo_input=archivo,
        parte="parte3",  
        resultado={
            "valor_objetivo": valor_objetivo,
            "ordenes_seleccionadas": list(ordenes),
            "pasillos_seleccionados": list(pasillos)
        }
    )