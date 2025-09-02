from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, lpSum, LpBinary
from leer_archivo import cargar_parametros_desde_archivo, imprimir_parametros
from clases import Parametros

def resolver_problema1(parametros: Parametros):

    print("\n======== RESOLVIENDO PROBLEMA 1 ========")
    
    modelo = LpProblem(name="problema1", sense=LpMaximize)

    # Variables binarias para seleccionar contenedor (solo uno)
    x_contenedor = [LpVariable(f"x_contenedor_{j}", cat=LpBinary) for j in range(parametros.total_contenedores)]
    
    # Variables binarias para seleccionar bolsitas
    y_bolsita = [LpVariable(f"y_bolsita_{j}", cat=LpBinary) for j in range(parametros.total_bolsitas)]
    
    # Restricción: seleccionar exactamente un contenedor
    modelo += lpSum(x_contenedor) == 1, "seleccionar_un_contenedor"
    
    # Restricciones de capacidad para cada tipo de ítem
    for i in range(parametros.total_items):
        # Cantidad disponible en el contenedor seleccionado
        disponible = lpSum(
            x_contenedor[contenedor.indice] * sum(item.cantidad for item in contenedor.items if item.tipo == i)
            for contenedor in parametros.contenedores
        )
        
        # Cantidad requerida por las bolsitas seleccionadas
        requerido = lpSum(
            y_bolsita[bolsita.indice] * sum(item.cantidad for item in bolsita.items if item.tipo == i)
            for bolsita in parametros.bolsitas
        )
        
        # La cantidad disponible debe ser mayor o igual a la requerida
        modelo += disponible >= requerido, f"capacidad_item_{i}"
    
    # Beneficio de los contenedores (beneficio fijo ba)
    beneficio_contenedores = lpSum(
        x_contenedor[contenedor.indice] * contenedor.beneficio  # contenedor.beneficio = ba
        for contenedor in parametros.contenedores
    )
    
    # Beneficio de los ítems de bolsitas (sin multiplicar por cantidad)
    beneficio_bolsitas = lpSum(
        y_bolsita[bolsita.indice] * sum(item.beneficio for item in bolsita.items)  # solo beneficio por ítem
        for bolsita in parametros.bolsitas
    )
    
    modelo += beneficio_contenedores + beneficio_bolsitas, "maximizar_beneficio_total"
    
    # Resolver modelo
    status = modelo.solve()
    
    # Mostrar resultados
    print(f"Status: {LpStatus[status]}")
    
    if status == 1:  # Optimal
        print("\nSOLUCIÓN ÓPTIMA ENCONTRADA:")
        
        # Contenedor seleccionado
        contenedor_seleccionado = None
        for j in range(parametros.total_contenedores):
            if x_contenedor[j].value() > 0.5:
                contenedor_seleccionado = j
                print(f"Contenedor seleccionado: {j}")
                break
        
        # Bolsitas seleccionadas
        bolsitas_seleccionadas = []
        for j in range(parametros.total_bolsitas):
            if y_bolsita[j].value() > 0.5:
                bolsitas_seleccionadas.append(j)
        
        print(f"Bolsitas seleccionadas: {bolsitas_seleccionadas}")
        print(f"Beneficio total: {modelo.objective.value()}")
        
        return {
            "contenedor": contenedor_seleccionado,
            "bolsitas": bolsitas_seleccionadas,
            "beneficio": modelo.objective.value()
        }
    else:
        print("No se encontró solución óptima.")
        return None

def resolver_problema2(parametros: Parametros):

    print("\n======== RESOLVIENDO PROBLEMA 2 ========")

    modelo = LpProblem(name="problema2", sense=LpMaximize)

    # Variables binarias para seleccionar contenedores (ahora pueden ser varios)
    x_contenedor = [LpVariable(f"x_contenedor_{j}", cat=LpBinary) for j in range(parametros.total_contenedores)]

    # Variables binarias para seleccionar bolsitas
    y_bolsita = [LpVariable(f"y_bolsita_{j}", cat=LpBinary) for j in range(parametros.total_bolsitas)]

    # Restricciones de capacidad por ítem
    for i in range(parametros.total_items):
        # Total disponible en los contenedores seleccionados
        disponible = lpSum(
            x_contenedor[contenedor.indice] * sum(item.cantidad for item in contenedor.items if item.tipo == i)
            for contenedor in parametros.contenedores
        )

        # Total requerido por las bolsitas seleccionadas
        requerido = lpSum(
            y_bolsita[bolsita.indice] * sum(item.cantidad for item in bolsita.items if item.tipo == i)
            for bolsita in parametros.bolsitas
        )

        # La cantidad disponible debe cubrir la requerida
        modelo += disponible >= requerido, f"capacidad_item_{i}"

    # Función objetivo: maximizar beneficios
    beneficio_contenedores = lpSum(
        x_contenedor[contenedor.indice] * contenedor.beneficio  # se toma el beneficio del contenedor directamente
        for contenedor in parametros.contenedores
    )

    beneficio_bolsitas = lpSum(
        y_bolsita[bolsita.indice] * sum(item.beneficio for item in bolsita.items)
        for bolsita in parametros.bolsitas
    )

    modelo += beneficio_contenedores + beneficio_bolsitas, "maximizar_beneficio_total"

    # Resolver modelo
    status = modelo.solve()
    print(f"Status: {LpStatus[status]}")

    if status == 1:  # Óptimo
        print("\nSOLUCIÓN ÓPTIMA ENCONTRADA:")

        contenedores_seleccionados = [
            j for j in range(parametros.total_contenedores) if x_contenedor[j].value() > 0.5
        ]
        bolsitas_seleccionadas = [
            j for j in range(parametros.total_bolsitas) if y_bolsita[j].value() > 0.5
        ]

        print(f"Contenedores seleccionados: {contenedores_seleccionados}")
        print(f"Bolsitas seleccionadas: {bolsitas_seleccionadas}")
        print(f"Beneficio total: {modelo.objective.value()}")

        return {
            "contenedores": contenedores_seleccionados,
            "bolsitas": bolsitas_seleccionadas,
            "beneficio": modelo.objective.value()
        }
    else:
        print("No se encontró solución óptima.")
        return None
    
if __name__ == "__main__":
    # Cargar parámetros desde el archivo
    nombre_archivo = "datos_de_entrada/input_0001.txt"
    parametros = cargar_parametros_desde_archivo(nombre_archivo)
    
    print(f"Cargando datos del archivo: {nombre_archivo}")
    imprimir_parametros(parametros)
    
    # Resolver problema 1
    solucion1 = resolver_problema1(parametros)
    
    # Resolver problema 2
    solucion2 = resolver_problema2(parametros)