from dataclasses import dataclass

@dataclass
class Item:
    tipo: int
    cantidad: int
    beneficio: int = 1

@dataclass
class Bolsita:
    indice: int
    items: list[Item]  # Lista de ítems (con cantidad y beneficio)

@dataclass
class Contenedor:
    indice: int
    items: list[Item]  # Lista de ítems (con cantidad y beneficio)
    beneficio: int = 1

@dataclass
class Parametros:
    total_items: int
    total_bolsitas: int
    total_contenedores: int
    bolsitas: list[Bolsita]
    contenedores: list[Contenedor]