import os

def _preparar_ruta_y_abrir(archivo_input, parte):
    nombre_dataset = os.path.basename(os.path.dirname(archivo_input))
    nombre_instancia = os.path.splitext(os.path.basename(archivo_input))[0]

    output_dir = os.path.join(parte, "OUTPUT", nombre_dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{nombre_instancia}.out")
    return open(output_path, "w")

def guardar_resultado_estandar(archivo_input, parte, status, modelo, x_vars=None, y_vars=None, n_ordenes=None, n_pasillos=None):
    with _preparar_ruta_y_abrir(archivo_input, parte) as f:
        if status in ["optimal", "optimal_inaccurate"]:
            f.write(f"Mejor valor objetivo: {modelo.getObjVal():.2f}\n")

            if x_vars is not None and n_ordenes is not None:
                ordenes_seleccionadas = [str(o) for o in range(n_ordenes) if modelo.getVal(x_vars[o]) > 0.5]
                f.write("Ordenes seleccionadas:\n")
                f.write(" ".join(ordenes_seleccionadas) + "\n")

            if y_vars is not None and n_pasillos is not None:
                pasillos_seleccionados = [str(a) for a in range(n_pasillos) if modelo.getVal(y_vars[a]) > 0.5]
                f.write("Pasillos seleccionados:\n")
                f.write(" ".join(pasillos_seleccionados) + "\n")

        else:
            f.write("No se encontró solución óptima.\n")

def guardar_resultado_simple(archivo_input, parte, status, modelo, x_vars, n_ordenes, pasillos_usados):
    with _preparar_ruta_y_abrir(archivo_input, parte) as f:
        if status in ["optimal", "optimal_inaccurate"]:
            f.write(f"Mejor valor objetivo: {int(modelo.getObjVal())}\n")

            ordenes_seleccionadas = [str(o) for o in range(n_ordenes) if modelo.getVal(x_vars[o]) > 0.5]
            f.write("Ordenes seleccionadas:\n")
            f.write(" ".join(ordenes_seleccionadas) + "\n")

            f.write("Pasillos utilizados:\n")
            f.write(" ".join(map(str, pasillos_usados)) + "\n")
        else:
            f.write("No se encontró solución óptima.\n")

def guardar_resultado(archivo_input, parte, resultado):
    with _preparar_ruta_y_abrir(archivo_input, parte) as f:
        f.write(f"Mejor valor objetivo: {resultado['valor_objetivo']}\n")
        f.write("Ordenes seleccionadas:\n")
        f.write(" ".join(map(str, resultado["ordenes_seleccionadas"])) + "\n")
        f.write("Pasillos seleccionados:\n")
        f.write(" ".join(map(str, resultado["pasillos_seleccionados"])) + "\n")
        
