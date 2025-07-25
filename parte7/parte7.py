import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from script import leer_config, ejecutar_todos_modelos, escribir_csv


def main():
    if len(sys.argv) < 2:
        print("Uso: python parte7.py archivo.cfg")
        cfg_path = "parte7/experimento.cfg"
    else:
        cfg_path = sys.argv[1]

    config = leer_config(cfg_path)
    metrica_dict, modelos = ejecutar_todos_modelos(config)

    csv_path = os.path.join(os.path.dirname(__file__), "resultados_parte7.csv")
    escribir_csv(metrica_dict, csv_path, modelos)
    print(f"\n✅ Resultados guardados en {csv_path}")

if __name__ == "__main__":
    main()
