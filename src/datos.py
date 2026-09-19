import io
import pandas as pd
import requests


def obtener_serie_ipc(
    serie_id: str = "101.1_I2NG_2016_M_22", fecha_inicio: str = "2017-01"):
    """Descarga y preprocesa la serie temporal del IPC desde la API oficial de Datos Argentina.

    Metadatos del ID ('101.1_I2NG_2016_M_22'):
        '101.1': Identificador del dataset de precios.
        'I2NG': Índice Nivel General.
        '2016': Año base de la serie (Base Diciembre 2016 = 100).
        'M': Frecuencia Mensual.
        '22': Cobertura geográfica (Nacional / Total País).

    Parámetros:
    serie_id : str, opcional
        ID oficial del indicador en el catálogo de series de tiempo de la API pública
        (por defecto '101.1_I2NG_2016_M_22').
    fecha_inicio : str, opcional
        Fecha inicial de consulta en formato 'AAAA-MM' (por defecto '2017-01').

    Retorna:
    DataFrame con el índice formateado como 'pd.DatetimeIndex'

    Excepciones:
    requests.exceptions.HTTPError
    Lanza una excepción si la API no responde con código HTTP 200 OK.
    """
    url_api = f"https://apis.datos.gob.ar/series/api/series/?ids={serie_id}&format=csv&start_date={fecha_inicio}"

    # User-Agent para identificarse formalmente frente al servidor del INDEC
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ActuarialAnalysis/1.0"
    }

    response = requests.get(url_api, headers=headers)
    response.raise_for_status()  # Valida que el status HTTP sea 200 OK

    # Carga y estructuración del DataFrame desde el flujo de texto devuelto
    df = pd.read_csv(io.StringIO(response.text))

    # Preprocesamiento de la dimensión temporal
    df["indice_tiempo"] = pd.to_datetime(df["indice_tiempo"])
    df.set_index("indice_tiempo", inplace=True)

    return df