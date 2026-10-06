"""Dashboard interactivo: Clasificación de Cobertura Forestal en el Roosevelt National Forest.

Gemelo interactivo del informe `jbook_ml_entregable2/Entregable2_Covertype.ipynb`.
- Pestaña 1: contexto geográfico y ecológico.
- Pestaña 2: análisis exploratorio (sección 3 del informe).
- Pestaña 3: evaluación y diagnóstico del modelo logístico (secciones 4 y 5).

Los resultados del modelo se leen de `data_dashboard/` (exportados por el informe). Las tablas
del EDA se calculan una sola vez a partir de `covertype.csv` con la misma partición del informe y
se guardan en `data_dashboard/eda_*.csv`, de modo que la aplicación arranca de inmediato.

Ejecución:  python app.py  ->  http://127.0.0.1:8050/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html
from dash_iconify import DashIconify

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "jbook_ml_entregable2"))
from _utils import (  # noqa: E402
    CLASS_NAMES, CLASS_PALETTE, CLASS_SHORT, DISPLAY_NAMES, NUM_COLS, RANDOM_STATE, WILD_NAMES,
    load_covertype, split_covertype, stratified_sample, wilderness_label,
)

DATA_DIR = ROOT / "data_dashboard"
BOOK_URL = "https://Maria856578.github.io/MachineLearning/"
CLASES = list(range(1, 8))
ETIQUETAS = [CLASS_SHORT[k] for k in CLASES]
ETIQUETA_A_ID = {CLASS_SHORT[k]: k for k in CLASES}
EMOJI = {1: "🌲", 2: "🌲", 3: "🌳", 4: "💧", 5: "🍂", 6: "🌲", 7: "❄️"}
AREA_COLOR = {"Rawah": "#1e88e5", "Neota": "#8e5ea2", "Comanche Peak": "#0f9d58", "Cache la Poudre": "#d9643a"}
AREAS = list(WILD_NAMES.values())
GRAFICO = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# --------------------------------------------------------------------------- datos del EDA
EDA = {
    "clases": DATA_DIR / "eda_clases.csv",
    "perfil": DATA_DIR / "eda_perfil_altitudinal.csv",
    "composicion": DATA_DIR / "eda_composicion_areas.csv",
    "areas": DATA_DIR / "eda_resumen_areas.csv",
    "variables": DATA_DIR / "eda_variables.csv",
    "correlacion": DATA_DIR / "eda_correlacion.csv",
    "solapamiento": DATA_DIR / "eda_solapamiento.csv",
    "muestra": DATA_DIR / "eda_muestra.csv",
}


def preparar_eda():
    """Calcula las tablas del EDA sobre el conjunto de entrenamiento del informe (semilla 42)."""
    print("Preparando tablas del EDA a partir de covertype.csv (solo la primera vez)...")
    _, X, y = load_covertype()
    X_train, _, y_train, _ = split_covertype(X, y)
    area_all = wilderness_label(X)

    conteo = y_train.value_counts().sort_index()
    pd.DataFrame({"id_clase": conteo.index, "celdas": conteo.values}).to_csv(EDA["clases"], index=False)

    perc = X_train.groupby(y_train)["Elevation"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    perc.columns = ["p05", "p25", "p50", "p75", "p95"]
    perc.rename_axis("id_clase").reset_index().to_csv(EDA["perfil"], index=False)

    comp = pd.crosstab(area_all, y, normalize="index").stack().rename("proporcion").reset_index()
    comp.columns = ["area", "id_clase", "proporcion"]
    comp.to_csv(EDA["composicion"], index=False)

    resumen = pd.DataFrame({
        "celdas": area_all.value_counts(),
        "elev_min": X.groupby(area_all)["Elevation"].min(),
        "elev_mediana": X.groupby(area_all)["Elevation"].median(),
        "elev_max": X.groupby(area_all)["Elevation"].max(),
        "dist_carreteras_mediana": X.groupby(area_all)["Horizontal_Distance_To_Roadways"].median(),
    }).rename_axis("area").reset_index()
    resumen.to_csv(EDA["areas"], index=False)

    corr = X_train[NUM_COLS].corr()
    corr.to_csv(EDA["correlacion"])
    vif = np.diag(np.linalg.inv(corr.to_numpy()))
    eta2 = []
    for c in NUM_COLS:
        x, media = X_train[c], X_train[c].mean()
        entre = sum(len(g) * (g.mean() - media) ** 2 for _, g in x.groupby(y_train))
        eta2.append(entre / ((x - media) ** 2).sum())
    pd.DataFrame({"variable": NUM_COLS, "eta2": eta2, "vif": vif}).to_csv(EDA["variables"], index=False)

    bordes = np.arange(1850, 3900, 20)
    hist = {k: np.histogram(X_train.loc[y_train == k, "Elevation"], bins=bordes)[0] / (y_train == k).sum() for k in CLASES}
    pares = [{"a": a, "b": b, "solapamiento": np.minimum(hist[a], hist[b]).sum()} for a in CLASES for b in CLASES if a < b]
    pd.DataFrame(pares).to_csv(EDA["solapamiento"], index=False)

    Xs, ys = stratified_sample(X_train, y_train, 40000, random_state=RANDOM_STATE)
    muestra = Xs[NUM_COLS].copy()
    muestra["area"] = wilderness_label(Xs).values
    muestra["id_clase"] = ys.values
    muestra.to_csv(EDA["muestra"], index=False)
    print("Tablas del EDA guardadas en data_dashboard/.")


if not all(p.exists() for p in EDA.values()):
    preparar_eda()

clases_df = pd.read_csv(EDA["clases"])
perfil_df = pd.read_csv(EDA["perfil"]).set_index("id_clase")
composicion_df = pd.read_csv(EDA["composicion"])
areas_df = pd.read_csv(EDA["areas"]).set_index("area")
variables_df = pd.read_csv(EDA["variables"]).set_index("variable")
correlacion_df = pd.read_csv(EDA["correlacion"], index_col=0)
solapamiento_df = pd.read_csv(EDA["solapamiento"])
muestra_df = pd.read_csv(EDA["muestra"])

# --------------------------------------------------------------------------- resultados del modelo
metricas_df = pd.read_csv(DATA_DIR / "metricas_comparativas.csv")
matriz_df = pd.read_csv(DATA_DIR / "matriz_confusion.csv")
coef_df = pd.read_csv(DATA_DIR / "importancia_coeficientes.csv")

CM = matriz_df.pivot(index="id_real", columns="id_predicha", values="conteo").loc[CLASES, CLASES].to_numpy()
CM_REAL = matriz_df.pivot(index="id_real", columns="id_predicha", values="prop_real").loc[CLASES, CLASES].to_numpy()
CM_PRED = matriz_df.pivot(index="id_real", columns="id_predicha", values="prop_predicha").loc[CLASES, CLASES].to_numpy()
RECALL = np.diag(CM_REAL)
PRECISION = np.diag(CM_PRED)

solapamiento_df["confusion"] = [CM_REAL[a - 1, b - 1] + CM_REAL[b - 1, a - 1] for a, b in zip(solapamiento_df.a, solapamiento_df.b)]
solapamiento_df["celdas"] = [CM[a - 1, b - 1] + CM[b - 1, a - 1] for a, b in zip(solapamiento_df.a, solapamiento_df.b)]
R_SOLAP = np.corrcoef(solapamiento_df["solapamiento"], solapamiento_df["confusion"])[0, 1]


def metrica(modelo, nombre, conjunto="prueba"):
    fila = metricas_df[(metricas_df.modelo == modelo) & (metricas_df.metrica == nombre) & (metricas_df.conjunto == conjunto)]
    return float(fila["valor"].iloc[0])


DUMMIES = ["Dummy (más frecuente)", "Dummy (uniforme)", "Dummy (estratificado)"]
LR = "Regresión logística"
ACC_LR, F1_LR = metrica(LR, "Accuracy"), metrica(LR, "F1 macro")
ACC_DUMMY = metrica("Dummy (más frecuente)", "Accuracy")
F1_DUMMY = max(metrica(d, "F1 macro") for d in DUMMIES)
BRECHA_F1 = metrica(LR, "F1 macro", "entrenamiento") - F1_LR

# Razón de odds por +100 m de elevación (efecto conjunto de las dos variables altitudinales; sección 5.4 del informe)
OR_100M = {1: 3.29, 2: 1.45, 3: 0.29, 4: 0.16, 5: 1.06, 6: 0.31, 7: 14.08}

# --------------------------------------------------------------------------- plantilla de gráficos
pio.templates["bosque"] = go.layout.Template(layout=dict(
    font=dict(family="Nunito, Segoe UI, sans-serif", size=13, color="#1f2d27"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fbfdfb",
    xaxis=dict(gridcolor="#e3ece5", zerolinecolor="#cfdcd3", linecolor="#cfdcd3"),
    yaxis=dict(gridcolor="#e3ece5", zerolinecolor="#cfdcd3", linecolor="#cfdcd3"),
    hoverlabel=dict(font=dict(family="Nunito, Segoe UI, sans-serif", size=13), bgcolor="white", bordercolor="#0f9d58"),
    title=dict(font=dict(size=16, color="#1b6f5c"), x=0.01),
    margin=dict(l=10, r=20, t=50, b=10),
    legend=dict(bgcolor="rgba(255,255,255,0.7)"),
))
pio.templates.default = "plotly_white+bosque"
DIVERGENTE = [[0, "#b5432a"], [0.25, "#f08a5d"], [0.5, "#fbf7ef"], [0.75, "#42a5f5"], [1, "#0d47a1"]]
SECUENCIAL = [[0, "#f1fbf3"], [0.15, "#b9e8c6"], [0.4, "#4fc38a"], [0.7, "#0f9d58"], [1, "#0b4f34"]]


# --------------------------------------------------------------------------- componentes reutilizables
def tarjeta(titulo, cuerpo, color="verde", icono=None, nota=None, seccion=None):
    cabecera = [DashIconify(icon=icono, width=22, className="me-2")] if icono else []
    contenido = list(cuerpo) if isinstance(cuerpo, (list, tuple)) else [cuerpo]
    if nota:
        enlace = [" ", html.A(f"Profundiza en la sección {seccion} del Jupyter Book →", href=BOOK_URL, target="_blank")] if seccion else []
        contenido.append(html.Div([html.B("💡 Interpretación: "), nota, *enlace], className="nota"))
    return dbc.Card([dbc.CardHeader(cabecera + [titulo], className=f"cab-{color} d-flex align-items-center"),
                     dbc.CardBody(contenido)], className="tarjeta")


def kpi(valor, etiqueta, detalle, estilo):
    return html.Div([html.Div(etiqueta, className="etiqueta"), html.Div(valor, className="valor"),
                     html.Div(detalle, className="detalle")], className=f"kpi kpi-{estilo}")


def fmt(x, d=3):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def miles(x):
    return f"{int(x):,}".replace(",", ".")


def badge_clase(k, **kw):
    return dbc.Badge(f"{EMOJI[k]} {CLASS_SHORT[k]}", style={"backgroundColor": CLASS_PALETTE[k]}, className="me-1 mb-1", **kw)


# --------------------------------------------------------------------------- figuras: contexto
def fig_perfil_altitudinal():
    pisos = [(1850, 2450, "Montano inferior", "rgba(224,165,38,0.18)"), (2450, 2750, "Montano superior", "rgba(76,154,42,0.16)"),
             (2750, 3250, "Subalpino", "rgba(27,111,92,0.16)"), (3250, 3900, "Límite arbóreo", "rgba(142,94,162,0.18)")]
    orden = perfil_df["p50"].sort_values().index
    fig = go.Figure()
    for lo, hi, nombre, color in pisos:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, line_width=0, layer="below",
                      annotation_text=f"<b>{nombre}</b>", annotation_position="top", annotation_font_color="#4f6159")
    for k in orden:
        p = perfil_df.loc[k]
        nombre = f"{EMOJI[k]} {CLASS_SHORT[k]}"
        hover = (f"<b>{CLASS_NAMES[k]}</b><br>Mediana: {miles(p.p50)} m<br>50 % central: {miles(p.p25)}–{miles(p.p75)} m"
                 f"<br>90 % central: {miles(p.p05)}–{miles(p.p95)} m<extra></extra>")
        fig.add_trace(go.Scatter(x=[p.p05, p.p95], y=[nombre] * 2, mode="lines", line=dict(color=CLASS_PALETTE[k], width=3),
                                 hovertemplate=hover, showlegend=False))
        fig.add_trace(go.Scatter(x=[p.p25, p.p75], y=[nombre] * 2, mode="lines", line=dict(color=CLASS_PALETTE[k], width=16),
                                 hovertemplate=hover, showlegend=False))
        fig.add_trace(go.Scatter(x=[p.p50], y=[nombre], mode="markers", hovertemplate=hover, showlegend=False,
                                 marker=dict(color="white", size=11, line=dict(color=CLASS_PALETTE[k], width=3))))
    fig.update_layout(height=400, xaxis_title="Elevación (m s. n. m.)", xaxis_range=[1850, 3900], margin=dict(t=40),
                      yaxis=dict(showgrid=False))
    return fig


def fig_composicion_areas():
    fig = go.Figure()
    for k in CLASES:
        sub = composicion_df[composicion_df.id_clase == k].set_index("area").reindex(AREAS[::-1]).fillna(0)
        fig.add_trace(go.Bar(y=sub.index, x=100 * sub["proporcion"], orientation="h", name=f"{EMOJI[k]} {CLASS_SHORT[k]}",
                             marker=dict(color=CLASS_PALETTE[k], line=dict(color="white", width=1.5)),
                             hovertemplate=f"<b>%{{y}}</b><br>{CLASS_NAMES[k]}: %{{x:.1f}} %<extra></extra>"))
    fig.update_layout(barmode="stack", height=330, xaxis_title="% de las celdas del área", xaxis_range=[0, 100],
                      legend=dict(orientation="h", y=-0.28, x=0), margin=dict(t=20))
    return fig


# --------------------------------------------------------------------------- figuras: EDA
NOTAS_VARIABLE = {
    "Elevation": "La elevación separa las coberturas en tres bloques: montano inferior (ponderosa, Douglas, álamo/sauce), montano superior (lodgepole, álamo temblón) y subalpino (pícea/abeto, krummholz). Es el predictor dominante del conjunto.",
    "Aspect": "La orientación por sí sola apenas discrimina (η² ≈ 0,005), pero separa al abeto Douglas, más frecuente en laderas frescas orientadas al norte, del pino ponderosa, propio de laderas cálidas.",
    "Slope": "Las coberturas del cañón del Poudre (ponderosa, Douglas, álamo/sauce) ocupan laderas más inclinadas, con suelos someros; las subalpinas se asientan sobre mesetas glaciares más suaves.",
    "Horizontal_Distance_To_Hydrology": "El álamo y el sauce son la única cobertura pegada al agua (mediana de 30 m): su nicho ribereño depende de un nivel freático somero.",
    "Vertical_Distance_To_Hydrology": "El bosque de galería se concentra en celdas casi al nivel del cauce (mediana de 6 m de desnivel). Fuera de esa franja, la variable apenas distingue coberturas.",
    "Horizontal_Distance_To_Roadways": "Las vías siguen los fondos de valle: las coberturas del piso bajo quedan más cerca de carreteras. Es un indicador regional de accesibilidad y presión antrópica.",
    "Hillshade_9am": "El abeto Douglas tiene la menor iluminación matinal: prefiere laderas en sombra, más frescas y húmedas.",
    "Hillshade_Noon": "El sombreado del mediodía depende sobre todo de la pendiente: las laderas abruptas reciben menos luz cenital.",
    "Hillshade_3pm": "El índice de la tarde refleja la orientación al oeste y es casi el espejo del de las 9 a. m. (r = −0,78): un bloque de iluminación redundante.",
    "Horizontal_Distance_To_Fire_Points": "Las coberturas montanas bajas están más cerca de puntos históricos de ignición, en coherencia con un régimen de incendios más frecuente en el piso bajo.",
}


def fig_desbalance(escala):
    df = clases_df.assign(nombre=lambda d: d.id_clase.map(lambda k: f"{EMOJI[k]} {CLASS_SHORT[k]}"))
    df["pct"] = 100 * df["celdas"] / df["celdas"].sum()
    df = df.sort_values("celdas")
    fig = go.Figure(go.Bar(
        y=df["nombre"], x=df["celdas"], orientation="h",
        marker=dict(color=[CLASS_PALETTE[k] for k in df.id_clase], line=dict(color="white", width=2)),
        text=[f"{miles(n)} ({fmt(p, 2)} %)" for n, p in zip(df.celdas, df.pct)], textposition="outside",
        customdata=df[["pct"]], hovertemplate="<b>%{y}</b><br>%{x:,} celdas<br>%{customdata[0]:.2f} % del entrenamiento<extra></extra>",
    ))
    razon = df.celdas.max() / df.celdas.min()
    fig.update_layout(height=380, title=f"Razón de desbalance {razon:.0f} : 1", xaxis_title="Celdas de entrenamiento",
                      xaxis_type="log" if escala == "log" else "linear", yaxis=dict(showgrid=False))
    if escala == "log":
        fig.update_xaxes(range=[3, 6.6])
    else:
        fig.update_xaxes(range=[0, df.celdas.max() * 1.35])
    return fig


def nombre_corto(c):
    return DISPLAY_NAMES[c].split(" (")[0]


ABREVIADO = {
    "Elevation": "Elevación", "Aspect": "Orientación", "Slope": "Pendiente",
    "Horizontal_Distance_To_Hydrology": "D. horiz. agua", "Vertical_Distance_To_Hydrology": "D. vert. agua",
    "Horizontal_Distance_To_Roadways": "D. carreteras", "Hillshade_9am": "Sombra 9 h", "Hillshade_Noon": "Sombra 12 h",
    "Hillshade_3pm": "Sombra 15 h", "Horizontal_Distance_To_Fire_Points": "D. incendios",
}


def fig_correlacion(vista):
    if vista == "vif":
        v = variables_df["vif"].sort_values()
        colores = ["#d9643a" if x > 10 else ("#f2b134" if x > 5 else "#0f9d58") for x in v]
        fig = go.Figure(go.Bar(y=[ABREVIADO[c] for c in v.index], x=v.values, orientation="h",
                               marker=dict(color=colores, line=dict(color="white", width=2)),
                               text=[fmt(x, 1) for x in v.values], textposition="outside",
                               hovertemplate="<b>%{y}</b><br>VIF = %{x:.1f}<extra></extra>"))
        for umbral, color in [(5, "#f2b134"), (10, "#d9643a")]:
            fig.add_vline(x=umbral, line_dash="dash", line_color=color, annotation_text=f"VIF {umbral}")
        fig.update_layout(height=430, xaxis_type="log", xaxis_title="Factor de inflación de la varianza (escala log)",
                          title="🟢 independiente · 🟡 moderado · 🔴 redundante", yaxis=dict(showgrid=False))
        return fig
    nombres = [ABREVIADO[c] for c in correlacion_df.columns]
    z = correlacion_df.to_numpy().copy()
    z[np.triu_indices_from(z, k=1)] = np.nan
    texto = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(z=z, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=DIVERGENTE, text=texto,
                               texttemplate="%{text}", textfont=dict(size=10), xgap=2, ygap=2,
                               colorbar=dict(title="r"), hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(height=470, yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(tickangle=35, showgrid=False),
                      title="Correlación de Pearson (triángulo inferior)")
    return fig


def fig_eta2():
    v = variables_df["eta2"].sort_values()
    colores = ["#0f9d58" if c == "Elevation" else "#9fd8b8" for c in v.index]
    fig = go.Figure(go.Bar(y=[nombre_corto(c) for c in v.index], x=v.values, orientation="h",
                           marker=dict(color=colores, line=dict(color="white", width=2)),
                           text=[fmt(x) for x in v.values], textposition="outside",
                           hovertemplate="<b>%{y}</b><br>η² = %{x:.3f}<br>(%{x:.1%} de su varianza explicada por la cobertura)<extra></extra>"))
    fig.update_layout(height=420, xaxis_range=[0, 0.72], xaxis_title="η² (proporción de varianza explicada por la cobertura)",
                      yaxis=dict(showgrid=False), title="🏔️ La elevación explica 6 veces más que la segunda variable")
    return fig


def fig_distribucion(variable, areas):
    sub = muestra_df[muestra_df.area.isin(areas)] if areas else muestra_df.iloc[0:0]
    orden = perfil_df["p50"].sort_values().index
    fig = go.Figure()
    for k in orden:
        valores = sub.loc[sub.id_clase == k, variable]
        if len(valores) < 5:
            continue
        fig.add_trace(go.Violin(y=valores, name=f"{EMOJI[k]} {CLASS_SHORT[k]}", line=dict(color=CLASS_PALETTE[k], width=1.5),
                                fillcolor=CLASS_PALETTE[k], opacity=0.78, box_visible=True, meanline_visible=False,
                                points=False, box=dict(width=0.22, fillcolor="white", line=dict(color="#333", width=1)),
                                hoveron="violins+kde", showlegend=False))
    if not fig.data:
        fig.add_annotation(text="Selecciona al menos un área silvestre", showarrow=False, font=dict(size=16))
        return fig.update_layout(height=440, xaxis=dict(visible=False), yaxis=dict(visible=False))

    # Geometría común a todas las variables: la silueta se limita al rango real de los datos (sin densidad
    # inventada fuera de él) y todos los violines comparten la misma anchura máxima.
    fig.update_traces(spanmode="hard", scalemode="width", width=0.6)

    # Rango dinámico del eje Y según la variable activa y las áreas filtradas: 12 % de holgura arriba;
    # abajo, la misma holgura si la variable admite negativos, o un margen mínimo (3 %) que despega la base
    # del borde sin mostrar valores negativos en variables acotadas en cero (distancias, pendiente, sombras).
    y_min, y_max = float(sub[variable].min()), float(sub[variable].max())
    recorrido = (y_max - y_min) or 1.0
    delta = 0.12 * recorrido
    inferior = y_min - (delta if y_min < 0 else 0.03 * recorrido)
    fig.update_layout(
        height=440, violingap=0.25, violingroupgap=0, xaxis=dict(showgrid=False),
        yaxis=dict(title=DISPLAY_NAMES[variable], range=[inferior, y_max + delta], automargin=True),
        title=f"{DISPLAY_NAMES[variable]} por cobertura · {miles(len(sub))} celdas de la muestra estratificada",
    )
    return fig


# --------------------------------------------------------------------------- figuras: modelo
METRICAS_VISTA = ["Accuracy", "Precisión macro", "Recall macro", "F1 macro"]


def fig_metricas(vista):
    fig = go.Figure()
    if vista == "ajuste":
        for conjunto, color in [("entrenamiento", "#26a6e8"), ("prueba", "#0f9d58")]:
            valores = [metrica(LR, m, conjunto) for m in METRICAS_VISTA]
            fig.add_trace(go.Bar(x=METRICAS_VISTA, y=valores, name=conjunto.capitalize(), marker_color=color,
                                 text=[fmt(v) for v in valores], textposition="outside",
                                 hovertemplate=f"<b>{conjunto.capitalize()}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"))
        fig.update_layout(title="Entrenamiento vs. prueba: brecha < 0,01 en todas las métricas")
    else:
        colores = {"Dummy (más frecuente)": "#ffcc80", "Dummy (uniforme)": "#ffab91", "Dummy (estratificado)": "#ce93d8", LR: "#0f9d58"}
        for modelo, color in colores.items():
            filas = metricas_df[(metricas_df.modelo == modelo) & (metricas_df.conjunto == "prueba")].set_index("metrica").loc[METRICAS_VISTA]
            error = None
            if modelo == LR:
                error = dict(type="data", array=filas.ic_superior - filas.valor, arrayminus=filas.valor - filas.ic_inferior,
                             color="#0b4f34", thickness=2, width=6)
            fig.add_trace(go.Bar(x=METRICAS_VISTA, y=filas.valor, name=modelo, marker_color=color, error_y=error,
                                 text=[fmt(v) for v in filas.valor] if modelo == LR else None, textposition="outside",
                                 hovertemplate=f"<b>{modelo}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>"))
        fig.update_layout(title="🎯 Regresión logística frente a la inferencia ingenua (prueba, 116.203 celdas)")
    fig.update_layout(barmode="group", height=420, yaxis_range=[0, 0.88], yaxis_title="Valor de la métrica",
                      legend=dict(orientation="h", y=-0.15, x=0), xaxis=dict(showgrid=False), bargap=0.25)
    return fig


def fig_confusion(modo):
    if modo == "conteo":
        z, texto, titulo, barra = CM, [[miles(v) for v in fila] for fila in CM], "Conteos absolutos", "celdas"
        hover = "Real: <b>%{y}</b><br>Predicha: <b>%{x}</b><br>%{z:,} celdas<extra></extra>"
        z_color = np.log10(CM + 1)
    else:
        m = CM_REAL if modo == "real" else CM_PRED
        z, z_color = m, m
        texto = [[f"{v:.2f}" for v in fila] for fila in m]
        titulo = "Normalizada por fila (diagonal = recall)" if modo == "real" else "Normalizada por columna (diagonal = precisión)"
        barra = "proporción"
        hover = "Real: <b>%{y}</b><br>Predicha: <b>%{x}</b><br>%{customdata:.1%}<extra></extra>"
    fig = go.Figure(go.Heatmap(
        z=z_color, x=ETIQUETAS, y=ETIQUETAS, customdata=z, text=texto, texttemplate="%{text}",
        textfont=dict(size=12), colorscale=SECUENCIAL, xgap=3, ygap=3, showscale=modo != "conteo",
        colorbar=dict(title=barra), zmin=0, zmax=None if modo == "conteo" else 1,
        hovertemplate=hover.replace("%{z:,}", "%{customdata:,}"),
    ))
    fig.update_layout(height=470, title=f"{titulo} · haz clic en una celda 👆", xaxis_title="Cobertura predicha",
                      yaxis_title="Cobertura real", yaxis=dict(autorange="reversed", showgrid=False),
                      xaxis=dict(tickangle=30, showgrid=False))
    return fig


def fig_solapamiento():
    df = solapamiento_df.copy()
    df["par"] = [f"{CLASS_SHORT[a]} ↔ {CLASS_SHORT[b]}" for a, b in zip(df.a, df.b)]
    destacados = df.nlargest(6, "confusion").index
    df["color"] = ["#d9643a" if i in destacados else "#26a6e8" for i in df.index]
    fig = go.Figure(go.Scatter(
        x=df.solapamiento, y=df.confusion, mode="markers+text",
        text=[p if i in destacados else "" for i, p in zip(df.index, df.par)],
        textposition=["top center" if (a, b) == (1, 2) else "middle left" if (a, b) == (1, 7) else "middle right"
                      for a, b in zip(df.a, df.b)],
        textfont=dict(size=10), customdata=np.stack([df.a, df.b, df.celdas, df.par], axis=1),
        marker=dict(size=10 + 30 * np.sqrt(df.celdas / df.celdas.max()), color=df.color, opacity=0.85,
                    line=dict(color="white", width=2)),
        hovertemplate="<b>%{customdata[3]}</b><br>Solapamiento altitudinal: %{x:.2f}<br>"
                      "Confusión bidireccional: %{y:.2f}<br>Celdas confundidas: %{customdata[2]:,}<extra></extra>",
    ))
    fig.update_layout(height=470, title=f"r = {fmt(R_SOLAP, 2)} entre nicho compartido y confusión · clic en un punto 👆",
                      xaxis_title="Solapamiento de elevación (0 = nada, 1 = idéntico)",
                      yaxis_title="Confusión bidireccional (a→b + b→a)", xaxis_range=[-0.32, 1.42],
                      xaxis_tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])
    return fig


GRUPOS = {"todas": None, "continua": "Continua", "area": "Área silvestre", "suelo": "Tipo de suelo"}


def fig_coeficientes(id_clase, grupo, n):
    sub = coef_df[coef_df.id_clase == id_clase]
    if GRUPOS[grupo]:
        sub = sub[sub.grupo == GRUPOS[grupo]]
    sub = sub.nlargest(n, "abs_coeficiente").sort_values("coeficiente")
    fig = go.Figure(go.Bar(
        y=sub.variable_legible, x=sub.coeficiente, orientation="h",
        marker=dict(color=["#0f9d58" if c > 0 else "#d9643a" for c in sub.coeficiente], line=dict(color="white", width=1.5)),
        customdata=np.stack([sub.odds_ratio, sub.grupo], axis=1),
        hovertemplate="<b>%{y}</b> (%{customdata[1]})<br>Coeficiente: %{x:+.2f}<br>Razón de odds: ×%{customdata[0]:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#4f6159", line_width=1)
    fig.update_layout(height=max(360, 30 * len(sub) + 120), yaxis=dict(showgrid=False),
                      title=f"{EMOJI[id_clase]} {CLASS_NAMES[id_clase]}: 🟢 aumenta · 🟠 reduce la probabilidad",
                      xaxis_title="Coeficiente (log-odds; continuas por desviación estándar)")
    return fig


NOTAS_COEF = {
    1: "Cada 100 m de ascenso multiplican sus odds por ≈ 3,3. Rawah y suelos subalpinos (9, 21, 22) la favorecen; Cache la Poudre y los suelos rocosos del cañón la excluyen.",
    2: "Efecto altitudinal moderado (≈ ×1,45 por cada 100 m): ocupa el centro del gradiente. Sus coeficientes más fuertes son edáficos (suelos 12 y 28 a favor; suelos 1, 4 y 5 del cañón en contra).",
    3: "La altitud la penaliza (≈ ×0,29 por cada 100 m). Los suelos 1 a 4, complejos de afloramiento rocoso del cañón, son su firma edáfica; Rawah la excluye.",
    4: "Cache la Poudre (+3,67) actúa como atajo geográfico: es la única área con bosque ribereño. Cada 100 m de ascenso reducen sus odds a ≈ 0,16, y la cercanía al agua la favorece.",
    5: "Rawah la favorece (+3,19) y Cache la Poudre la excluye (−3,51). Sin variables de perturbación, ninguna predictora le da suficiente evidencia: su recall es apenas 0,04.",
    6: "Los índices de sombreado tienen signos opuestos (+2,94 a las 3 p. m., −1,74 al mediodía) por colinealidad: solo su efecto conjunto tiene lectura física, a favor de laderas poco expuestas.",
    7: "El efecto altitudinal más fuerte del modelo: +100 m multiplican sus odds por ≈ 14. Los suelos pedregosos de alta montaña (4, 37, 39) completan su firma.",
}


def explicacion_confusion(a, b):
    """Texto didáctico para la celda (real a, predicha b) o el par {a, b}."""
    if a == b:
        return [html.H6(f"{EMOJI[a]} {CLASS_NAMES[a]}: celda diagonal (aciertos)"),
                html.P([f"Recall = {fmt(RECALL[a - 1])} · precisión = {fmt(PRECISION[a - 1])}. ",
                        "Haz clic fuera de la diagonal para entender por qué el modelo confunde dos coberturas."])]
    par = frozenset((a, b))
    fila = solapamiento_df[(solapamiento_df.a == min(a, b)) & (solapamiento_df.b == max(a, b))].iloc[0]
    datos = (f"Solapamiento altitudinal = {fmt(fila.solapamiento, 2)} · {CLASS_SHORT[a]} → {CLASS_SHORT[b]}: "
             f"{fmt(100 * CM_REAL[a - 1, b - 1], 1)} % · {CLASS_SHORT[b]} → {CLASS_SHORT[a]}: {fmt(100 * CM_REAL[b - 1, a - 1], 1)} % · "
             f"{miles(fila.celdas)} celdas confundidas en total.")
    textos = {
        frozenset((1, 2)): ("🔥 Transición sucesional",
                            "El pino lodgepole coloniza los sitios tras incendios de reemplazo y, con los siglos, la pícea y el abeto lo sustituyen bajo su dosel. "
                            "Entre 2.800 y 3.300 m el mismo terreno puede sostener cualquiera de las dos: decide el tiempo desde el último incendio, que no está en los datos. "
                            "El modelo traza en la práctica un umbral cercano a 3.050 m (pícea/abeto mal clasificado: mediana de 2.981 m)."),
        frozenset((1, 7)): ("🏔️ Ecotono del límite arbóreo",
                            "El krummholz es la forma achaparrada de las mismas especies del bosque subalpino, moldeada por el viento y el hielo. "
                            "Lo determinan la exposición y la nieve a escalas menores que la celda de 30 m, y es 10 veces menos frecuente: la duda se resuelve a favor de la pícea/abeto."),
        frozenset((3, 6)): ("🧭 Partición por orientación",
                            "Es el par con mayor solapamiento altitudinal. El abeto Douglas ocupa laderas frescas orientadas al norte (54 % de sus celdas) y el ponderosa, laderas cálidas. "
                            "El efecto de la orientación depende de la pendiente (interacción no lineal) y está repartido entre índices de sombreado colineales."),
        frozenset((3, 4)): ("💧 Nicho ribereño no monótono",
                            "El álamo/sauce se concentra en celdas casi al nivel del cauce (|Δh| < 5 m), y por encima y por debajo desaparece: un pico que una recta no representa. "
                            "Además, el corredor ribereño es compartido: incluso junto al agua el pino ponderosa ocupa ≈ 40 % de las celdas."),
        frozenset((4, 6)): ("💧 Corredor ribereño compartido",
                            "El álamo/sauce comparte con el abeto Douglas la elevación, el área (Cache la Poudre) y buena parte de los suelos; la capa hidrológica no registra el nivel freático ni las crecidas que delimitan el bosque de galería."),
        frozenset((2, 5)): ("🍂 Cobertura definida por la perturbación",
                            "El álamo temblón es clonal y rebrota tras incendios: su presencia depende de la historia del sitio, no de la topografía. "
                            "Su nicho altitudinal está contenido en el del lodgepole, 30 veces más frecuente, y el modelo casi nunca lo elige (recall 0,04)."),
    }
    titulo, cuerpo = textos.get(par, ("📏 Vecinos en el gradiente",
                                      "Estas coberturas comparten parte de su franja altitudinal; cuanto mayor es el solapamiento, más se confunden. "
                                      "Los pares con solapamiento inferior a 0,2 apenas registran errores."))
    return [html.H6(f"{titulo}: {CLASS_SHORT[a]} ↔ {CLASS_SHORT[b]}"), html.P(cuerpo), html.Small(datos, className="text-muted")]


# --------------------------------------------------------------------------- pestaña 1: contexto
ESPECIES = [
    (1, "Picea engelmannii · Abies lasiocarpa", "Subalpino", "Bosque de clímax fresco y húmedo, con nieve prolongada; especies tolerantes a la sombra."),
    (2, "Pinus contorta var. latifolia", "Montano superior y subalpino", "Pionera con conos serótinos que se abren con el fuego; forma rodales tras incendios."),
    (3, "Pinus ponderosa var. scopulorum", "Montano inferior", "Laderas secas y cálidas orientadas al sur; corteza gruesa resistente al fuego."),
    (4, "Populus angustifolia · Salix spp.", "Fondos de valle", "Bosque de galería ribereño ligado a un nivel freático somero."),
    (5, "Populus tremuloides", "Montano y subalpino", "Especie clonal que rebrota de raíz tras perturbaciones."),
    (6, "Pseudotsuga menziesii var. glauca", "Montano inferior y medio", "Prefiere laderas frescas orientadas al norte."),
    (7, "Picea, Abies y Pinus flexilis achaparrados", "Límite arbóreo", "Árboles deformados por el viento y el hielo en el borde superior del bosque."),
]
AREAS_INFO = {
    "Rawah": ("🏞️", "Montes Medicine Bow: valles glaciares y más de 25 lagos. Matriz subalpina de pícea/abeto y lodgepole.", "Exposición media"),
    "Neota": ("🏔️", "Meseta subalpina junto al paso Cameron, casi toda sobre 2.950 m. La más alta y con más krummholz.", "Exposición baja"),
    "Comanche Peak": ("⛰️", "Cordillera Mummy, contigua al Parque Nacional: del piso montano a la tundra alpina.", "Exposición media"),
    "Cache la Poudre": ("💧", "Cañón bajo del río: ponderosa, Douglas y la única ribera de álamo/sauce.", "Exposición alta"),
}


def tarjeta_area(area):
    emoji, texto, exposicion = AREAS_INFO[area]
    r = areas_df.loc[area]
    return html.Div([
        html.H5(f"{emoji} {area}"),
        dbc.Badge(exposicion, color="light", text_color="dark", className="mb-2"),
        html.Div(texto, className="mb-2"),
        html.Div([html.B(f"{miles(r.celdas)} celdas"), f" · {miles(r.elev_min)}–{miles(r.elev_max)} m"]),
        html.Div(f"Elevación mediana {miles(r.elev_mediana)} m · carretera a {miles(r.dist_carreteras_mediana)} m"),
    ], className="area", style={"background": f"linear-gradient(135deg, {AREA_COLOR[area]}, {AREA_COLOR[area]}bb)"})


def tarjeta_especie(k, cientifico, piso, nicho):
    return html.Div([
        html.H6(f"{EMOJI[k]} {CLASS_NAMES[k]}"),
        html.Div(cientifico, className="cientifico mb-1"),
        dbc.Badge(piso, style={"backgroundColor": CLASS_PALETTE[k]}, className="mb-2"),
        html.Div(nicho),
    ], className="especie", style={"borderTopColor": CLASS_PALETTE[k]})


pestana_contexto = html.Div([
    dbc.Row([
        dbc.Col(kpi("581.012", "🗺️ Celdas de 30 × 30 m", "sin valores faltantes", 1), md=True, xs=6, className="mb-3"),
        dbc.Col(kpi("54", "📐 Predictoras", "10 continuas + 44 indicadoras", 2), md=True, xs=6, className="mb-3"),
        dbc.Col(kpi("7", "🌲 Coberturas", "clases a predecir", 3), md=True, xs=6, className="mb-3"),
        dbc.Col(kpi("4", "🏞️ Áreas silvestres", "Roosevelt National Forest", 5), md=True, xs=6, className="mb-3"),
        dbc.Col(kpi("≈ 2.000 m", "🏔️ Gradiente altitudinal", "de 1.859 a 3.858 m s. n. m.", 4), md=True, xs=12, className="mb-3"),
    ], className="g-3"),
    html.Img(src="/assets/banner_montana.svg", className="banner mb-4", alt="Pisos altitudinales de las Montañas Rocosas"),
    dbc.Row([
        dbc.Col(tarjeta("¿Qué problema resolvemos?", [
            html.P(["Predecir la ", html.B("cobertura forestal dominante"), " de cada celda del terreno a partir solo de su ",
                    html.B("cartografía"), ": elevación, orientación, pendiente, distancias al agua, a carreteras y a puntos de incendio, área silvestre y tipo de suelo."]),
            html.P(["Los inventarios de campo son costosos en zonas remotas; un modelo cartográfico apoya la ",
                    html.B("planificación forestal, la evaluación del riesgo de incendios y la conservación"), "."]),
            html.Div([dbc.Badge("🎯 Clasificación multiclase", color="success", className="me-1 mb-1"),
                      dbc.Badge("⚖️ Desbalance 103 : 1", color="warning", className="me-1 mb-1"),
                      dbc.Badge("🤖 Regresión logística multinomial", color="info", className="me-1 mb-1")]),
        ], color="lago", icono="mdi:target"), lg=4, className="mb-3"),
        dbc.Col(tarjeta("Gradiente altitudinal: cada especie en su piso", dcc.Graph(figure=fig_perfil_altitudinal(), config=GRAFICO),
                        color="verde", icono="mdi:image-filter-hdr",
                        nota="La temperatura baja ≈ 6 °C por cada 1.000 m de ascenso y cada cobertura ocupa la franja que tolera. "
                             "Las barras gruesas de clases vecinas se solapan: esas transiciones serán las zonas de duda del modelo.",
                        seccion="1.4"), lg=8, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(tarjeta("Composición de cada área silvestre", dcc.Graph(figure=fig_composicion_areas(), config=GRAFICO),
                        color="azul", icono="mdi:map-marker-radius",
                        nota="Cada área ocupa un tramo distinto del gradiente. Cache la Poudre es otro mundo: sin pícea/abeto ni krummholz, "
                             "y es la única con bosque ribereño de álamo/sauce.",
                        seccion="1.2"), lg=5, className="mb-3"),
        dbc.Col([
            html.H4("🏞️ Las cuatro áreas silvestres", className="seccion-titulo"),
            dbc.Row([dbc.Col(tarjeta_area(a), md=6, className="mb-3") for a in AREAS]),
        ], lg=7),
    ]),
    html.H4("🌲 Las siete coberturas forestales", className="seccion-titulo mt-2"),
    dbc.Row([dbc.Col(tarjeta_especie(*e), xl=3, lg=4, md=6, className="mb-3") for e in ESPECIES]),
    html.Div(dbc.Button([DashIconify(icon="mdi:book-open-page-variant", width=22, className="me-2"),
                         "Leer el informe completo en Jupyter Book"], href=BOOK_URL, target="_blank",
                        className="boton-libro mt-2"), className="text-center"),
])

# --------------------------------------------------------------------------- pestaña 2: EDA
pestana_eda = html.Div([
    dbc.Row([
        dbc.Col(tarjeta("Desbalance de clases", [
            dbc.RadioItems(id="escala-desbalance", inline=True, value="log",
                           options=[{"label": " Escala logarítmica", "value": "log"}, {"label": " Escala lineal", "value": "lineal"}]),
            dcc.Graph(id="fig-desbalance", config=GRAFICO),
        ], color="ocre", icono="mdi:scale-unbalanced",
            nota="El lodgepole y la pícea/abeto forman la matriz subalpina (87 % de las celdas); el álamo/sauce es un corredor ribereño estrecho. "
                 "El modelo tenderá a favorecer las clases frecuentes, por eso se usa F1 macro y validación estratificada.",
            seccion="3.1"), lg=6, className="mb-3"),
        dbc.Col(tarjeta("Poder discriminante de cada variable", dcc.Graph(figure=fig_eta2(), config=GRAFICO),
                        color="verde", icono="mdi:podium-gold",
                        nota="η² mide qué parte de la variabilidad de una variable se explica por la cobertura. La elevación (0,62) domina; "
                             "orientación, sombreado e hidrología solo separan pares concretos de especies.",
                        seccion="3.3"), lg=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(tarjeta("Distribución de las variables por cobertura", [
            dbc.Row([
                dbc.Col([html.Div("🔎 Variable cartográfica", className="etiqueta-control"),
                         dcc.Dropdown(id="variable-distribucion", value="Elevation", clearable=False,
                                      options=[{"label": DISPLAY_NAMES[c], "value": c} for c in NUM_COLS])], md=5),
                dbc.Col([html.Div("🏞️ Áreas silvestres", className="etiqueta-control"),
                         dbc.Checklist(id="areas-distribucion", value=AREAS, inline=True,
                                       options=[{"label": f" {a}", "value": a} for a in AREAS])], md=7),
            ], className="mb-2"),
            dcc.Graph(id="fig-distribucion", config=GRAFICO),
            html.Div(id="nota-distribucion", className="nota"),
        ], color="lavanda", icono="mdi:chart-bell-curve"), lg=6, className="mb-3"),
        dbc.Col(tarjeta("Correlación y multicolinealidad", [
            dbc.RadioItems(id="vista-correlacion", inline=True, value="matriz",
                           options=[{"label": " Matriz de correlación", "value": "matriz"}, {"label": " VIF", "value": "vif"}]),
            dcc.Graph(id="fig-correlacion", config=GRAFICO),
        ], color="azul", icono="mdi:grid",
            nota="Los tres índices de sombreado se calculan con la orientación y la pendiente: son redundantes (VIF de 38 a 144). "
                 "La elevación aporta información casi independiente (VIF 1,3).",
            seccion="3.2"), lg=6, className="mb-3"),
    ]),
])

# --------------------------------------------------------------------------- pestaña 3: modelo
pestana_modelo = html.Div([
    dbc.Row([
        dbc.Col(kpi(fmt(ACC_LR), "🎯 Accuracy en prueba", f"vs. {fmt(ACC_DUMMY)} del clasificador mayoritario", 1), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi(fmt(F1_LR), "⚖️ F1 macro en prueba", f"vs. {fmt(F1_DUMMY)} del mejor dummy", 2), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi(f"+{fmt(100 * (F1_LR / F1_DUMMY - 1), 0)} %", "🚀 Ganancia en F1 macro", "sobre la inferencia ingenua", 3), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi(fmt(BRECHA_F1, 4), "🩺 Brecha entrenamiento–prueba", "F1 macro: sin sobreajuste", 4), lg=3, md=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(tarjeta("Desempeño frente a la línea base", [
            dbc.RadioItems(id="vista-metricas", inline=True, value="base",
                           options=[{"label": " 🎯 Modelo vs. dummies", "value": "base"},
                                    {"label": " 🩺 Diagnóstico de ajuste", "value": "ajuste"}]),
            dcc.Graph(id="fig-metricas", config=GRAFICO),
        ], color="verde", icono="mdi:trophy",
            nota="Predecir siempre lodgepole acierta el 49 % por puro desbalance, pero su F1 macro es 0,09. El modelo cuadruplica el F1 macro, "
                 "y entrenamiento y prueba casi coinciden: su límite es el sesgo de un modelo lineal, no la falta de datos.",
            seccion="4"), lg=6, className="mb-3"),
        dbc.Col(tarjeta("¿Qué pesa en cada cobertura? Coeficientes", [
            dbc.Row([
                dbc.Col([html.Div("🌲 Cobertura", className="etiqueta-control"),
                         dcc.Dropdown(id="clase-coef", value=7, clearable=False,
                                      options=[{"label": f"{EMOJI[k]} {CLASS_NAMES[k]}", "value": k} for k in CLASES])], md=6),
                dbc.Col([html.Div("🧩 Tipo de variable", className="etiqueta-control"),
                         dcc.Dropdown(id="grupo-coef", value="todas", clearable=False,
                                      options=[{"label": "Todas", "value": "todas"}, {"label": "Continuas", "value": "continua"},
                                               {"label": "Áreas silvestres", "value": "area"}, {"label": "Tipos de suelo", "value": "suelo"}])], md=6),
            ]),
            html.Div("🔢 Número de variables", className="etiqueta-control mt-2"),
            dcc.Slider(id="n-coef", min=5, max=20, step=1, value=10, marks={5: "5", 10: "10", 15: "15", 20: "20"}),
            dcc.Graph(id="fig-coef", config=GRAFICO),
            html.Div(id="nota-coef", className="nota"),
        ], color="lavanda", icono="mdi:tune-variant"), lg=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(tarjeta("Matriz de confusión", [
            dbc.RadioItems(id="modo-confusion", inline=True, value="real",
                           options=[{"label": " % de la clase real (recall)", "value": "real"},
                                    {"label": " % de la predicción (precisión)", "value": "pred"},
                                    {"label": " Conteos", "value": "conteo"}]),
            dcc.Graph(id="fig-confusion", config=GRAFICO),
        ], color="terracota", icono="mdi:grid-large",
            nota="Los errores no se dispersan: se concentran entre coberturas vecinas del gradiente. "
                 "El álamo temblón casi desaparece (91 % → lodgepole) y el abeto Douglas se confunde con el ponderosa.",
            seccion="5.1"), lg=6, className="mb-3"),
        dbc.Col(tarjeta("Los errores siguen el nicho compartido", dcc.Graph(id="fig-solapamiento", figure=fig_solapamiento(), config=GRAFICO),
                        color="lago", icono="mdi:vector-intersection",
                        nota="Cada punto es un par de coberturas: cuanto más comparten su franja de elevación, más las confunde el modelo. "
                             "El tamaño indica cuántas celdas se confundieron.",
                        seccion="5.2"), lg=6, className="mb-3"),
    ]),
    dbc.Card([dbc.CardHeader([DashIconify(icon="mdi:magnify", width=22, className="me-2"),
                              "Explorador de confusiones: la ecología detrás de cada error"], className="cab-ocre d-flex align-items-center"),
              dbc.CardBody(html.Div(id="explicacion-confusion", className="panel-explicacion"))], className="tarjeta mb-3"),
])

# --------------------------------------------------------------------------- aplicación
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap"],
    title="Cobertura Forestal · Roosevelt National Forest",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div([
    html.Div(dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🌲 Clasificación de Cobertura Forestal - Roosevelt National Forest"),
                html.Div("Montañas Rocosas del norte de Colorado · Curso Machine Learning · Segundo entregable", className="subtitulo"),
                html.Div([dbc.Badge("👩‍🔬 Ana Galván", color="light", text_color="success"),
                          dbc.Badge("👩‍🔬 Valeria Oñate", color="light", text_color="primary"),
                          dbc.Badge("👩‍🔬 María José Ruiz", color="light", text_color="danger")], className="mt-2"),
            ], lg=9),
            dbc.Col(dbc.Button([DashIconify(icon="mdi:book-open-variant", width=20, className="me-2"), "Jupyter Book"],
                               href=BOOK_URL, target="_blank", className="boton-libro"),
                    lg=3, className="d-flex align-items-center justify-content-lg-end mt-3 mt-lg-0"),
        ]),
    ], fluid="xl"), className="hero mb-4"),
    dbc.Container([
        dcc.Location(id="url"),
        dbc.Tabs([
            dbc.Tab(html.Div(pestana_contexto, className="contenido-tab"), label="🌲 Contexto y Territorio", tab_id="contexto"),
            dbc.Tab(html.Div(pestana_eda, className="contenido-tab"), label="📊 Análisis Exploratorio", tab_id="eda"),
            dbc.Tab(html.Div(pestana_modelo, className="contenido-tab"), label="🤖 Machine Learning Model", tab_id="modelo"),
        ], id="pestanas", active_tab="contexto"),
        html.Div(["Datos: Covertype, UCI Machine Learning Repository (Blackard y Dean, 1998; CC BY 4.0) · ",
                  html.A("Informe técnico en Jupyter Book", href=BOOK_URL, target="_blank")], className="pie"),
    ], fluid="xl"),
])


@app.callback(Output("pestanas", "active_tab"), Input("url", "search"))
def abrir_pestana(busqueda):
    """Permite enlazar directamente a una pestaña: /?tab=eda o /?tab=modelo."""
    pestana = (busqueda or "").replace("?tab=", "")
    return pestana if pestana in {"contexto", "eda", "modelo"} else "contexto"


@app.callback(Output("fig-desbalance", "figure"), Input("escala-desbalance", "value"))
def actualizar_desbalance(escala):
    return fig_desbalance(escala)


@app.callback(Output("fig-correlacion", "figure"), Input("vista-correlacion", "value"))
def actualizar_correlacion(vista):
    return fig_correlacion(vista)


@app.callback(Output("fig-distribucion", "figure"), Output("nota-distribucion", "children"),
              Input("variable-distribucion", "value"), Input("areas-distribucion", "value"))
def actualizar_distribucion(variable, areas):
    eta = variables_df.loc[variable, "eta2"]
    nota = [html.B("💡 Interpretación: "), NOTAS_VARIABLE[variable], f" (η² = {fmt(eta)}). ",
            html.A("Profundiza en la sección 3.3 del Jupyter Book →", href=BOOK_URL, target="_blank")]
    return fig_distribucion(variable, areas), nota


@app.callback(Output("fig-metricas", "figure"), Input("vista-metricas", "value"))
def actualizar_metricas(vista):
    return fig_metricas(vista)


@app.callback(Output("fig-coef", "figure"), Output("nota-coef", "children"),
              Input("clase-coef", "value"), Input("grupo-coef", "value"), Input("n-coef", "value"))
def actualizar_coeficientes(id_clase, grupo, n):
    nota = [html.B("💡 Interpretación: "), NOTAS_COEF[id_clase],
            f" Razón de odds por +100 m de elevación: ×{fmt(OR_100M[id_clase], 2)}. ",
            html.A("Profundiza en la sección 5.4 del Jupyter Book →", href=BOOK_URL, target="_blank")]
    return fig_coeficientes(id_clase, grupo, n), nota


@app.callback(Output("fig-confusion", "figure"), Input("modo-confusion", "value"))
def actualizar_confusion(modo):
    return fig_confusion(modo)


@app.callback(Output("explicacion-confusion", "children"),
              Input("fig-confusion", "clickData"), Input("fig-solapamiento", "clickData"))
def explicar_confusion(clic_matriz, clic_puntos):
    from dash import ctx
    if ctx.triggered_id == "fig-solapamiento" and clic_puntos:
        a, b = (int(v) for v in clic_puntos["points"][0]["customdata"][:2])
        return explicacion_confusion(a, b)
    if ctx.triggered_id == "fig-confusion" and clic_matriz:
        p = clic_matriz["points"][0]
        return explicacion_confusion(ETIQUETA_A_ID[p["y"]], ETIQUETA_A_ID[p["x"]])
    return explicacion_confusion(1, 2)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
