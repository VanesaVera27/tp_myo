from clases import Item, Bolsita, Contenedor, Parametros

# Formato del archivo de entrada:
# - Primera línea: o (bolsitas), i (tipos de ítems), a (contenedores)

# - Siguientes o líneas: cada una indica una bolsita:
#     k seguido de k pares (tipo_item, cantidad)

# - Siguientes a líneas: cada una indica un contenedor:
#     l seguido de l pares (tipo_item, cantidad)

# Se asume beneficio = 1 para todos los ítems


def cargar_parametros_desde_archivo(path: str) -> Parametros:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    o, i, a = map(int, lines[0].split())

    bolsitas = []
    for idx, line in enumerate(lines[1:1 + o]):
        parts = list(map(int, line.split()))
        k = parts[0]
        items = [Item(tipo=parts[j], cantidad=parts[j + 1]) for j in range(1, 2 * k + 1, 2)]
        bolsitas.append(Bolsita(indice=idx, items=items))

    contenedores = []
    for idx, line in enumerate(lines[1 + o:1 + o + a]):
        parts = list(map(int, line.split()))
        l = parts[0]
        items = [Item(tipo=parts[j], cantidad=parts[j + 1]) for j in range(1, 2 * l + 1, 2)]
        contenedores.append(Contenedor(indice=idx, items=items))

    return Parametros(
        total_bolsitas=o,
        total_items=i,
        total_contenedores=a,
        bolsitas=bolsitas,
        contenedores=contenedores
    )

def imprimir_parametros(parametros):
    print("================== BOLSITAS ==================")
    for b in parametros.bolsitas:
        print(f"Bolsita {b.indice}:")
        for item in b.items:
            print(f"  Item tipo {item.tipo}, cantidad {item.cantidad}, beneficio {item.beneficio}")

    print("\n================== CONTENEDORES ==================")
    for c in parametros.contenedores:
        print(f"Contenedor {c.indice}:")
        for item in c.items:
            print(f"  Item tipo {item.tipo}, cantidad {item.cantidad}, beneficio {item.beneficio}")


# Ejemplo de uso
if __name__ == "__main__":
    parametros = cargar_parametros_desde_archivo("input_0001.txt")
    imprimir_parametros(parametros)