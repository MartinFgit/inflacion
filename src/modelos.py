import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


def test_adf_dinamico(serie, alpha=0.05):
    """Evalúa el test ADF seleccionando dinámicamente el modelo determinístico
    finitamente correcto basado en pruebas de significancia de los cuales:
    
    Modelo 3: con intercepto y tendencia deterministica
    Modelo 2: con intercepto
    Modelo 1: sin ningun componente deterministico

    
    """
    s_clean = serie.dropna()

    # 1. Probar Modelo 3 (Constante + Tendencia)
    adf_ct = adfuller(s_clean, regression="ct", autolag="AIC")

    # Estimación OLS auxiliar para evaluar la significancia de la tendencia (β_t)
    lag_opt = adf_ct[2] #rezagos optimos
    dy = s_clean.diff().iloc[lag_opt + 1 :]
    y_lag = s_clean.shift(1).iloc[lag_opt + 1 :]

    X_ct = pd.DataFrame(
        {
            "y_lag": y_lag,
            "const": 1,
            "trend": range(1, len(y_lag) + 1),
        }
    )
    for i in range(1, lag_opt + 1):
        X_ct[f"dy_lag_{i}"] = s_clean.diff().shift(i).iloc[lag_opt + 1 :]

    ols_ct = sm.OLS(dy, X_ct).fit()
    p_val_trend = ols_ct.pvalues.get("trend", 1.0)

    # Si la tendencia es significativa (p < alpha), el Modelo 3 es el correcto
    if p_val_trend < alpha:
        return {
            "modelo": "Modelo 3 (Constante + Tendencia)",
            "adf_stat": adf_ct[0],
            "p_value": adf_ct[1],
            "estacionaria": adf_ct[1] < alpha,
            "lags": lag_opt,
        }

    # 2. Probar Modelo 2 (Solo Constante / Drift)
    adf_c = adfuller(s_clean, regression="c", autolag="AIC")

    # REGLA ECONOMÉTRICA: Si la media de la serie es distinta de cero,
    # el Modelo 2 es obligatorio (no se descarta la constante).
    # Se evalúa si el promedio es estadísticamente diferente de 0.
    mean_val = s_clean.mean()
    std_err = s_clean.std() / np.sqrt(len(s_clean))
    t_stat_mean = abs(mean_val / std_err) if std_err > 0 else 0

    # Si el promedio no es cero (t_stat > 1.96) o la constante OLS es significativa:
    if t_stat_mean > 1.96:
        return {
            "modelo": "Modelo 2 (Solo Constante/Drift)",
            "adf_stat": adf_c[0],
            "p_value": adf_c[1],
            "estacionaria": adf_c[1] < alpha,
            "lags": adf_c[2],
        }

    # 3. Solo si la serie oscila estrictamente alrededor de 0 se usa el Modelo 1
    adf_n = adfuller(s_clean, regression="n", autolag="AIC")
    return {
        "modelo": "Modelo 1 (Sin componentes determinísticos)",
        "adf_stat": adf_n[0],
        "p_value": adf_n[1],
        "estacionaria": adf_n[1] < alpha,
        "lags": adf_n[2],
    }
    
    
def determinar_orden_integracion(
    serie, nombre_serie="Serie", max_d=3, alpha=0.05
):
    """Aplica iterativamente el test ADF dinámico sobre la serie en niveles y sus

    sucesivas diferencias hasta encontrar la estacionariedad.

    Determina si la serie es I(0), I(1), I(2), ..., I(max_d).
    """
    s_eval = serie.copy()
    d = 0

    print(f"==================================================")
    print(f" EVALUACIÓN DE ORDEN DE INTEGRACIÓN I(d): {nombre_serie}")
    print(f"==================================================\n")

    while d <= max_d:
        res = test_adf_dinamico(s_eval, alpha=alpha)

        d_str = "Niveles (d=0)" if d == 0 else f"Diferencia d={d}"
        print(f"--- Evaluando {d_str} ---")
        print(f"   Espec. Seleccionada: {res['modelo']}")
        print(
            f"   Estadístico ADF: {res['adf_stat']:.4f} | p-value: {res['p_value']:.4f}"
        )

        if res["estacionaria"]:
            print(
                f"\n--> CONCLUSIÓN: La serie '{nombre_serie}' es ESTACIONARIA en d={d}."
            )
            print(
                f"    Pasa la prueba con especificación: {res['modelo']} (p-value = {res['p_value']:.4f} < {alpha})"
            )
            print(
                f"    ORDEN DE INTEGRACIÓN FINAL: I({d}) -> Usar d = {d} en ARIMA."
            )
            return {"d": d, "detalles": res}
        else:
            print(
                f"   Result: NO es estacionaria en d={d} (p-value = {res['p_value']:.4f} >= {alpha}).\n"
            )
            d += 1
            s_eval = s_eval.diff().dropna()

    print(
        f"ATENCIÓN: La serie no alcanzó la estacionariedad luego de {max_d} diferencias."
    )
    return {"d": None, "detalles": None}


def seleccionar_especificacion_arima(serie, p_range=range(0, 4), d_range=[1], q_range=range(0, 4)):
    """
    Evalúa múltiples especificaciones ARIMA sobre una serie de tiempo.
    Guarda el objeto del modelo estimado en la columna 'model_fit'.
    """
    resultados = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(serie, order=(p,d,q))
                        model_fit = model.fit()
                        
                        #booleana de convergencia
                        convergio = bool(model_fit.mle_retvals.get('converged', False))
                        
                        resultados.append({
                            'orden': f"ARIMA({p},{d},{q})",
                            'p': p, 'd': d, 'q': q,
                            'BIC': model_fit.bic,
                            'AIC': model_fit.aic,
                            'convergencia': convergio,
                            'model_fit': model_fit  # Se guarda el objeto entrenado
                        })
                    except Exception as e:
                        resultados.append({
                            'orden': f"ARIMA({p},{d},{q})",
                            'p': p, 'd': d, 'q': q,
                            'BIC': None, 'AIC': None,
                            'convergencia': False,
                            'model_fit': None,
                            'msj_error': str(e)
                        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by='BIC', ascending=True).reset_index(drop=True)
    
    return df_resultados


def evaluar_jarque_bera(residuos, nombre_modelo):
    """Ejecución del test de Jarque-Bera"""
    jb_stat, jb_pvalue = stats.jarque_bera(residuos)
    
    print(f"==================================================")
    print(f"Test de Jarque-Bera: {nombre_modelo}")
    print(f"==================================================")
    print(f"Estadístico JB : {jb_stat:.4f}")
    print(f"p-valor        : {jb_pvalue:.4f}")