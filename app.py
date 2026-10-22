import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Segmentación de Clientes",
    page_icon="🛍️",
    layout="wide"
)

st.title("Segmentación de Clientes con K-Means")
st.write(
    "Aplicación interactiva de minería de datos "
    "para analizar y clasificar clientes."
)

# ==========================================
# CARGAR MODELO Y DATOS
# ==========================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo/kmeans_clientes.pkl")
    scaler = joblib.load("modelo/scaler.pkl")
    return modelo, scaler

@st.cache_data
def cargar_datos():
    clientes = pd.read_csv(
        "modelo/clientes_segmentados.csv"
    )

    centroides = pd.read_csv(
        "modelo/centroides.csv"
    )

    codo = pd.read_csv(
        "modelo/metricas_codo.csv"
    )

    return clientes, centroides, codo

# Cargar archivos
modelo, scaler = cargar_modelo()

clientes, centroides, codo = cargar_datos()

# Mostrar información general
st.subheader("Resumen general")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total de clientes",
        len(clientes)
    )

with col2:
    st.metric(
        "Número de clusters",
        modelo.n_clusters
    )

with col3:
    st.metric(
        "Variables utilizadas",
        modelo.n_features_in_
    )

# ==========================================
# MÉTODO DEL CODO INTERACTIVO
# ==========================================

st.divider()

st.subheader("Método del Codo")

st.write(
    "Explora cómo cambia la inercia según "
    "el número de clusters utilizados."
)

# Selector interactivo de K
k_seleccionado = st.slider(
    "Selecciona un número de clusters",
    min_value=int(codo["K"].min()),
    max_value=int(codo["K"].max()),
    value=5,
    step=1
)

# Obtener la inercia correspondiente
inercia_seleccionada = codo.loc[
    codo["K"] == k_seleccionado,
    "Inercia"
].iloc[0]

# Crear la gráfica
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    codo["K"],
    codo["Inercia"],
    marker="o",
    linewidth=2,
    label="Inercia"
)

# Resaltar el K seleccionado
ax.scatter(
    k_seleccionado,
    inercia_seleccionada,
    color="red",
    s=160,
    zorder=5,
    label=f"K = {k_seleccionado}"
)

ax.set_title("Método del Codo - K-Means")
ax.set_xlabel("Número de clusters (K)")
ax.set_ylabel("Inercia (WCSS)")
ax.set_xticks(codo["K"])
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# Mostrar el valor seleccionado
st.metric(
    "Inercia para el K seleccionado",
    f"{inercia_seleccionada:.4f}"
)

st.info(
    "El modelo final fue entrenado con K=5, "
    "seleccionado mediante el método del codo "
    "y el coeficiente de silueta."
)

# ==========================================
# VISUALIZACIÓN INTERACTIVA DE CLUSTERS
# ==========================================

import plotly.express as px
import plotly.graph_objects as go

st.divider()

st.subheader("Segmentación de clientes con K-Means")

st.write(
    "Explora los cinco segmentos identificados "
    "mediante el algoritmo K-Means."
)

# Controles interactivos
col1, col2 = st.columns(2)

with col1:
    clusters_disponibles = sorted(
        clientes["Cluster"].unique().tolist()
    )

    clusters_seleccionados = st.multiselect(
        "Selecciona los clusters que deseas visualizar",
        options=clusters_disponibles,
        default=clusters_disponibles
    )

with col2:
    mostrar_centroides = st.checkbox(
        "Mostrar centroides",
        value=True
    )

# Filtrar clientes
clientes_filtrados = clientes[
    clientes["Cluster"].isin(clusters_seleccionados)
].copy()

# Convertir clusters a texto para la leyenda
clientes_filtrados["Segmento"] = (
    clientes_filtrados["Cluster"]
    .apply(lambda x: f"Cluster {x}")
)

# Crear gráfico interactivo
fig_clusters = px.scatter(
    clientes_filtrados,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segmento",
    hover_data=[
        "CustomerID",
        "Age",
        "Gender",
        "Perfil"
    ],
    title="Segmentación interactiva de clientes",
    labels={
        "Annual Income (k$)": "Ingreso anual (miles USD)",
        "Spending Score (1-100)": "Puntuación de gasto"
    }
)

# Agregar centroides reales
if mostrar_centroides:

    centroides_filtrados = centroides[
        centroides["Cluster"].isin(
            clusters_seleccionados
        )
    ]

    fig_clusters.add_trace(
        go.Scatter(
            x=centroides_filtrados["Annual Income (k$)"],
            y=centroides_filtrados["Spending Score (1-100)"],
            mode="markers+text",
            marker=dict(
                symbol="x",
                size=16,
                color="black",
                line=dict(width=2, color="white")
            ),
            text=[
                f"Centroide {k}"
                for k in centroides_filtrados["Cluster"]
            ],
            textposition="top center",
            name="Centroides"
        )
    )

# Configuración visual
fig_clusters.update_layout(
    height=600,
    legend_title="Segmentos",
    hovermode="closest"
)

st.plotly_chart(
    fig_clusters,
    use_container_width=True
)

# Información de los clientes filtrados
st.metric(
    "Clientes visualizados",
    len(clientes_filtrados)
)

# Tabla interactiva
st.subheader("Detalle de clientes")

st.dataframe(
    clientes_filtrados,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# CLASIFICACION DE NUEVOS CLIENTES
# ==========================================

st.divider()
st.subheader("Clasificacion de nuevos clientes")

st.write(
    "Ingresa los datos de un cliente para determinar "
    "su segmento utilizando el modelo K-Means entrenado."
)

perfiles = {
    0: "Clientes promedio",
    1: "Clientes de alto valor potencial",
    2: "Gasto alto e ingresos bajos",
    3: "Clientes con potencial de crecimiento",
    4: "Clientes de bajo consumo"
}

estrategias = {
    0: "Programas de puntos, promociones y fidelizacion.",
    1: "Beneficios exclusivos y atencion personalizada.",
    2: "Descuentos selectivos y paquetes accesibles.",
    3: "Campanas personalizadas y activacion comercial.",
    4: "Promociones economicas y ofertas de entrada."
}

with st.form("formulario_cliente"):

    ingreso = st.number_input(
        "Ingreso anual (miles de USD)",
        min_value=0.0,
        max_value=500.0,
        value=60.0,
        step=1.0
    )

    gasto = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

    clasificar = st.form_submit_button(
        "Clasificar cliente",
        type="primary"
    )

if clasificar:

    nuevo_cliente = pd.DataFrame({
        "Annual Income (k$)": [ingreso],
        "Spending Score (1-100)": [gasto]
    })

    # Aplicar el escalador original
    cliente_escalado = scaler.transform(nuevo_cliente)

    # Clasificacion real mediante K-Means
    cluster_predicho = int(
        modelo.predict(cliente_escalado)[0]
    )

    perfil = perfiles[cluster_predicho]
    estrategia = estrategias[cluster_predicho]

    st.success("Clasificacion realizada correctamente")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Cluster asignado", cluster_predicho)

    with col2:
        st.metric(
            "Clientes del segmento",
            int((clientes["Cluster"] == cluster_predicho).sum())
        )

    st.write("**Perfil comercial:**", perfil)
    st.write("**Estrategia recomendada:**", estrategia)

    resultado = nuevo_cliente.copy()
    resultado["Cluster"] = cluster_predicho
    resultado["Perfil"] = perfil

    st.dataframe(
        resultado,
        hide_index=True,
        use_container_width=True
    )

    # Visualizar la posicion del nuevo cliente
    fig_prediccion = px.scatter(
        clientes,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=clientes["Cluster"].astype(str),
        opacity=0.55,
        title="Ubicacion del nuevo cliente"
    )

    fig_prediccion.add_trace(
        go.Scatter(
            x=[ingreso],
            y=[gasto],
            mode="markers",
            marker=dict(
                symbol="star",
                size=20,
                color="black",
                line=dict(
                    color="yellow",
                    width=2
                )
            ),
            name="Nuevo cliente"
        )
    )

    st.plotly_chart(
        fig_prediccion,
        use_container_width=True
    )

# ==========================================
# ANALISIS DEL COEFICIENTE DE SILUETA
# ==========================================

st.divider()
st.subheader("Evaluacion del modelo: coeficiente de silueta")

st.write(
    "El coeficiente de silueta permite evaluar "
    "la separacion y cohesion de los clusters. "
    "Cuanto mayor sea el valor, mejor sera la "
    "separacion de los grupos."
)

# Resultados obtenidos durante el entrenamiento
resultados_silueta = pd.DataFrame({
    "K": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Silueta": [
        0.3213,
        0.4666,
        0.4939,
        0.5547,
        0.5399,
        0.5281,
        0.4567,
        0.4571,
        0.4362
    ]
})

fig_silueta = px.line(
    resultados_silueta,
    x="K",
    y="Silueta",
    markers=True,
    title="Coeficiente de silueta para diferentes valores de K"
)

fig_silueta.add_vline(
    x=5,
    line_dash="dash",
    line_color="red",
    annotation_text="K optimo = 5"
)

fig_silueta.update_layout(
    xaxis_title="Numero de clusters (K)",
    yaxis_title="Coeficiente de silueta",
    xaxis=dict(dtick=1),
    height=450
)

st.plotly_chart(
    fig_silueta,
    use_container_width=True
)

mejor_k = int(
    resultados_silueta.loc[
        resultados_silueta["Silueta"].idxmax(),
        "K"
    ]
)

mejor_silueta = float(
    resultados_silueta["Silueta"].max()
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Mejor numero de clusters", mejor_k)

with col2:
    st.metric(
        "Mejor coeficiente de silueta",
        f"{mejor_silueta:.4f}"
    )

st.info(
    "K=5 obtuvo el mayor coeficiente de silueta "
    "entre los valores evaluados, respaldando "
    "la seleccion del modelo final."
)

# ==========================================
# ANALISIS COMERCIAL DE LOS SEGMENTOS
# ==========================================

st.divider()
st.subheader("Analisis comercial de los cinco segmentos")

st.write(
    "Los clusters identificados permiten agrupar "
    "clientes con comportamientos similares y "
    "proponer estrategias comerciales diferenciadas."
)

resumen_segmentos = (
    clientes.groupby("Cluster")
    .agg(
        Clientes=("CustomerID", "count"),
        Ingreso_promedio=("Annual Income (k$)", "mean"),
        Gasto_promedio=("Spending Score (1-100)", "mean")
    )
    .reset_index()
)

resumen_segmentos["Porcentaje"] = (
    resumen_segmentos["Clientes"]
    / len(clientes) * 100
)

resumen_segmentos["Perfil"] = (
    resumen_segmentos["Cluster"].map(perfiles)
)

resumen_segmentos["Estrategia"] = (
    resumen_segmentos["Cluster"].map(estrategias)
)

resumen_segmentos = resumen_segmentos.rename(
    columns={
        "Cluster": "Cluster",
        "Clientes": "Cantidad de clientes",
        "Ingreso_promedio": "Ingreso promedio (k$)",
        "Gasto_promedio": "Spending Score promedio",
        "Porcentaje": "Porcentaje (%)",
        "Perfil": "Perfil comercial",
        "Estrategia": "Estrategia recomendada"
    }
)

st.dataframe(
    resumen_segmentos.round(2),
    use_container_width=True,
    hide_index=True
)

# Distribucion porcentual de clientes
fig_distribucion = px.bar(
    resumen_segmentos,
    x="Perfil comercial",
    y="Cantidad de clientes",
    color="Cluster",
    title="Distribucion de clientes por segmento",
    text="Cantidad de clientes"
)

fig_distribucion.update_layout(
    xaxis_title="Segmento comercial",
    yaxis_title="Numero de clientes",
    height=500
)

st.plotly_chart(
    fig_distribucion,
    use_container_width=True
)

# Descargar resultados
st.subheader("Descargar resultados de segmentacion")

csv_resultados = clientes.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Descargar clientes segmentados (CSV)",
    data=csv_resultados,
    file_name="clientes_segmentados_kmeans.csv",
    mime="text/csv"
)

st.caption(
    "Las estrategias comerciales son recomendaciones "
    "basadas en los perfiles identificados por K-Means."
)
