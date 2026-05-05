#!/usr/bin/env python3
"""
Main script para el proyecto de Gaming Behavior Machine Learning
Integra entrenamiento y predicción del modelo de engagement
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_engagement_model import main as train_main
from src.predict_engagement import main as predict_main


def menu_interactivo():
    print("\n" + "="*50)
    print("GAMING BEHAVIOR MACHINE LEARNING")
    print("="*50)
    print("1. Entrenar modelo")
    print("2. Hacer predicción con jugador de ejemplo")
    print("3. Entrenar y predecir (pipeline completo)")
    print("4. Iniciar API servidor web")
    print("5. Salir")
    print("="*50)
    
    while True:
        try:
            opcion = input("\nSelecciona una opción (1-5): ").strip()
            if opcion in ["1", "2", "3", "4", "5"]:
                return opcion
            else:
                print("Opción no válida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nSaliendo del programa...")
            sys.exit(0)


def entrenar_modelo():
    print("\nIniciando entrenamiento del modelo...")
    print("-" * 40)
    try:
        train_main()
        print("\nEntrenamiento completado exitosamente!")
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        return False
    return True


def hacer_prediccion():
    print("\nRealizando predicción con jugador de ejemplo...")
    print("-" * 50)
    try:
        predict_main()
        print("\nPredicción completada!")
    except Exception as e:
        print(f"\nError durante la predicción: {e}")
        print("Asegúrate de haber entrenado el modelo primero (opción 1)")
        return False
    return True


def iniciar_api():
    import sys
    
    from src.config import settings
    
    print("\nIniciando servidor API...")
    print("-" * 40)
    print(f"API estará disponible en: http://{settings.host}:{settings.port}")
    print(f"Documentación: http://{settings.host}:{settings.port}/docs")
    print(f"Rate limit: {settings.rate_limit}")
    if settings.valid_api_keys:
        print("Autenticación: API key requerida")
    print("-" * 40)
    print("Presiona Ctrl+C para detener el servidor")
    print("="*40)
    
    try:
        import uvicorn
        
        uvicorn.run(
            "src.api:app",
            host=settings.host,
            port=settings.port,
            reload=True,
        )
        
    except ImportError:
        print("\nError: uvicorn no está instalado")
        print("Ejecuta: pip install uvicorn")
        return False
    except KeyboardInterrupt:
        print("\n\nServidor API detenido")
        return True
    except Exception as e:
        print(f"\nError al iniciar API: {e}")
        return False
    
    return True


def pipeline_completo():
    print("\nEjecutando pipeline completo (entrenamiento + predicción)...")
    print("-" * 60)
    
    if not entrenar_modelo():
        return False
    
    print("\n" + "="*60)
    
    if not hacer_prediccion():
        return False
    
    print("\nPipeline completo finalizado exitosamente!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Main script para Gaming Behavior Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                    # Menú interactivo
  python main.py --train           # Entrenar modelo
  python main.py --predict         # Hacer predicción
  python main.py --pipeline        # Pipeline completo
  python main.py --api             # Iniciar servidor API
        """
    )
    
    parser.add_argument("--train", action="store_true", 
                       help="Entrenar el modelo")
    parser.add_argument("--predict", action="store_true", 
                       help="Hacer predicción con ejemplo")
    parser.add_argument("--pipeline", action="store_true", 
                       help="Ejecutar pipeline completo (entrenar + predecir)")
    parser.add_argument("--api", action="store_true", 
                       help="Iniciar servidor API")
    
    args = parser.parse_args()
    
    if not any([args.train, args.predict, args.pipeline, args.api]):
        opcion = menu_interactivo()
        
        if opcion == "1":
            entrenar_modelo()
        elif opcion == "2":
            hacer_prediccion()
        elif opcion == "3":
            pipeline_completo()
        elif opcion == "4":
            iniciar_api()
        elif opcion == "5":
            print("Adiós!")
            return
    else:
        if args.api:
            iniciar_api()
        elif args.pipeline:
            pipeline_completo()
        elif args.train:
            entrenar_modelo()
        elif args.predict:
            hacer_prediccion()


if __name__ == "__main__":
    main()
