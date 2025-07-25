#!/usr/bin/env python3
"""
Script para verificar la factibilidad y optimalidad de una solución
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cargar_input import leer_input

def verificar_solucion(archivo_input, pasillos_sel, ordenes_sel, valor_obj_reportado):
    """
    Verifica si una solución es factible y calcula su valor objetivo real
    """
    W, S, LB, UB = leer_input(archivo_input)
    
    print(f"🔍 VERIFICANDO SOLUCIÓN")
    print(f"📂 Archivo: {archivo_input}")
    print(f"📊 Instancia: {len(W)} órdenes, {len(S)} pasillos, LB={LB}, UB={UB}")
    print(f"🎯 Valor objetivo reportado: {valor_obj_reportado}")
    print()
    
    # 1. Verificar que los índices son válidos
    if max(pasillos_sel) >= len(S):
        print("❌ ERROR: Índice de pasillo inválido")
        return False
    
    if max(ordenes_sel) >= len(W):
        print("❌ ERROR: Índice de orden inválido")
        return False
    
    # 2. Calcular capacidad total disponible
    capacidad_total = [0] * len(W[0])  # Asumiendo que todas las órdenes tienen la misma cantidad de ítems
    
    for p in pasillos_sel:
        for i in range(len(S[p])):
            capacidad_total[i] += S[p][i]
    
    print(f"📦 Capacidad total disponible (primeros 5 ítems): {capacidad_total[:5]}")
    
    # 3. Calcular demanda total de las órdenes seleccionadas
    demanda_total = [0] * len(W[0])
    valor_objetivo_real = 0
    
    for o in ordenes_sel:
        valor_objetivo_real += sum(W[o])  # Beneficio de la orden
        for i in range(len(W[o])):
            demanda_total[i] += W[o][i]
    
    print(f"📊 Demanda total (primeros 5 ítems): {demanda_total[:5]}")
    print(f"💰 Valor objetivo calculado: {valor_objetivo_real}")
    
    # 4. Verificar restricciones de capacidad
    factible = True
    for i in range(len(capacidad_total)):
        if demanda_total[i] > capacidad_total[i]:
            print(f"❌ VIOLACIÓN en ítem {i}: demanda={demanda_total[i]} > capacidad={capacidad_total[i]}")
            factible = False
    
    # 5. Verificar límites LB y UB
    if valor_objetivo_real < LB:
        print(f"❌ Viola límite inferior: {valor_objetivo_real} < {LB}")
        factible = False
    
    if valor_objetivo_real > UB:
        print(f"❌ Viola límite superior: {valor_objetivo_real} > {UB}")
        factible = False
    
    # 6. Verificar concordancia con valor reportado
    if valor_objetivo_real != valor_obj_reportado:
        print(f"⚠️ DISCREPANCIA: Calculado={valor_objetivo_real}, Reportado={valor_obj_reportado}")
    
    # Resultado final
    if factible and valor_objetivo_real == valor_obj_reportado:
        print(f"✅ SOLUCIÓN VÁLIDA Y FACTIBLE")
        print(f"📈 Utilización de límite superior: {valor_objetivo_real}/{UB} = {100*valor_objetivo_real/UB:.1f}%")
        if valor_objetivo_real == UB:
            print(f"🎯 ¡SOLUCIÓN ÓPTIMA! (alcanza el límite superior)")
        return True
    else:
        print(f"❌ SOLUCIÓN INVÁLIDA")
        return False

if __name__ == "__main__":
    # Datos de la nueva ejecución (con instancia reducida)
    archivo = "../datos_de_entrada/B/instance_0001.txt"
    pasillos = [4, 10, 15, 9, 29, 12, 26, 2, 3, 17, 8, 23, 7, 25, 16, 22, 21, 13, 0, 24, 27, 11, 14, 20, 28, 1, 18, 19, 5, 6]
    ordenes = [651, 594, 124, 298, 517, 657, 491, 255, 269, 356, 500, 840, 118, 139, 142, 397, 409, 628, 658, 660, 676, 677, 775, 906, 910, 938, 957, 964]
    valor = 350
    
    verificar_solucion(archivo, pasillos, ordenes, valor)
