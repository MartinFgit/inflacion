# Análisis Econométrico y Modelado ARIMA de la Inflación en Argentina (IPC)

Este repositorio contiene un proyecto para el modelado, selección de especificaciones y evaluación de predicción del Índice de Precios al Consumidor (IPC) Nivel General de Argentina (Base 2016), utilizando series temporales e inferencia de Cuasi-Máxima Verosimilitud (QMLE).


## Funciones Reutilizables del Proyecto

* Cargado de Datos: descarga directa desde la API de Datos Abiertos de Argentina (`apis.datos.gob.ar`) de la serie oficial del IPC Nivel General (Base 2016) mediante el módulo `src/datos.py`.
* Corrida del Test de Raíz Unitaria: evalúa la presencia de una raiz unitaria bajo distintas especificaciones(intercepto y tendencia) de la ecuacion de Dickey-Fuller chequeando la significancia de sus coeficientes a travéz de la funcion `test_adf_dinamico` del modulo `src/datos.py`
* Determinación del Orden de Integración: evalúa secuencialmente la serie en niveles y en sus sucesivas diferencias mediante pruebas de raíces unitarias hasta alcanzar la estacionariedad, identificando el número de diferenciaciones requeridas d a través de la función `determinar_orden_integracion` del módulo `src/datos.py`
* Búsqueda en Grilla y Selección ARIMA: itera sobre distintas especificaciones, de un rango de parametros definidos, del modelo ARIMA, aplicando filtrado explícito de convergencia de Máxima Verosimilitud (MLE) y ordenando las especificaciones según el criterio de informacion bayesiano (BIC) a través de la función `seleccionar_especificacion_arima` del módulo `src/modelos.py`
* Diagnóstico de Normalidad Residual: Testea las propiedades de asimetría y curtosis de los residuos del modelo frente a una distribución teórica gaussiana (test de Jarque-Bera) mediante la función `evaluar_jarque_bera` del módulo `src/modelos.py`

## Estructura del Repositorio

```text
inflacion/
│
├── .gitignore                      # Exclusiones de Git
├── README.md                       # Portada del proyecto
├── requirements.txt                # Dependencias del entorno
│
├── 01_estacionariedad_y_descomposicion.ipynb  
├── 02_estimacion_y_seleccion_arima.ipynb      
├── 03_pronostico_y_evaluacion.ipynb           
│
└── src/                            # Módulo reutilizable de Python
    ├── __init__.py
    ├── datos.py                     
    └── modelos.py                  
```


## Instalación y Replicabilidad
Para ejecutar este proyecto de forma local, sigue estos pasos:

1. Clonar el repositorio
Abre la terminal o Git Bash y ejecuta:
git clone https://github.com/TU-USUARIO-GITHUB/inflacion.git
cd inflacion

2. Crear y activar un entorno virtual
En Windows:
python -m venv venv
venv\Scripts\activate

En macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Instalar dependencias
pip install -r requirements.txt

4. Ejecución
Abre VS Code u otro entorno en la carpeta raíz y ejecuta los cuadernos en orden secuencial:

01_estacionariedad_y_descomposicion.ipynb

02_estimacion_y_seleccion_arima.ipynb

03_pronostico_y_evaluacion.ipynb