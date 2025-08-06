from columns_solver import Columns  
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cargar_input import leer_input
from guardar_output import guardar_resultado

# Cargar datos desde archivo
archivo_input = "datos_de_entrada/b/instance_0002.txt"
W, S, LB, UB = leer_input(archivo_input)

# Instanciar y ejecutar
solver = Columns(W, S, LB, UB)

resultado = solver.Opt_ExplorarCantidadPasillos(150)
if resultado:
    print("=== RESULTADO FINAL ===")
    print("Valor objetivo:", resultado["valor_objetivo"])
    print("Pasillos seleccionados:", resultado["pasillos_seleccionados"])
    print("Órdenes seleccionadas:", resultado["ordenes_seleccionadas"])
else:
    print("No se encontró solución con el umbral de tiempo dado.")


# === Guardar resultados en archivo .out ===

guardar_resultado(
    archivo_input=archivo_input,
    parte="parte5",  
    resultado={
        "valor_objetivo": resultado["valor_objetivo"],
        "ordenes_seleccionadas": list(resultado["ordenes_seleccionadas"]),
        "pasillos_seleccionados": list(resultado["pasillos_seleccionados"])
    }
)