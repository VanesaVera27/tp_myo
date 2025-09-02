def leer_input(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        lineas = f.read().strip().split('\n')
    
    o, i, a = map(int, lineas[0].split())

    W = [[0] * i for _ in range(o)]
    for idx in range(1, 1 + o):
        partes = list(map(int, lineas[idx].split()))
        k = partes[0]
        for j in range(k):
            item = partes[1 + 2*j]
            cantidad = partes[1 + 2*j + 1]
            W[idx - 1][item] = cantidad

    S = [[0] * i for _ in range(a)]
    for idx in range(1 + o, 1 + o + a):
        partes = list(map(int, lineas[idx].split()))
        l = partes[0]
        for j in range(l):
            item = partes[1 + 2*j]
            cantidad = partes[1 + 2*j + 1]
            S[idx - (1 + o)][item] = cantidad

    LB, UB = map(int, lineas[-1].split())
    return W, S, LB, UB
    
