import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Configuración de estilo oscuro con acentos rojos
def set_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1a1a1a;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ff4b4b;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #262730;
        }
        .st-bh {
            background-color: #262730;
        }
        .st-cg {
            background-color: #ff4b4b;
        }
        .st-cx {
            background-color: #ff4b4b;
        }
        .css-1aumxhk {
            background-color: #0E1117;
            background-image: none;
            color: white;
        }
        .css-1v3fvcr {
            color: white;
        }
        .css-1q8dd3e {
            color: white;
        }
        .stAlert {
            background-color: #262730;
        }
        .st-bq {
            border-color: #ff4b4b;
        }
        .stSelectbox label, .stMultiselect label, .stCheckbox label, .stRadio label {
            color: white !important;
        }
        .stTextInput label, .stNumberInput label, .stTextArea label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_dark_theme()

# Vista Inicial - Presentación
def show_welcome():
    st.title("🏠 Análisis de Airbnb en Dallas, Texas")
    st.markdown("---")

    # Aplicar estilo CSS para imágenes (ahora en el tema oscuro)

    # Imagen de Dallas
    st.image(r"C:\Users\leirb\Downloads\puebas\img\dallas.jpg", 
             use_container_width=True,  # Cambiado a use_container_width
             caption="Vista panorámica de Dallas, Texas")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🌇 Dallas, Texas")
        st.markdown("""
        - 3ra ciudad más grande de Texas
        - Centro económico y cultural
        - Clima subtropical húmedo
        - Atracciones principales:
          * Reunion Tower
          * Dallas Arboretum
          * Sixth Floor Museum
        """)

    with col2:
        st.subheader("🏡 Sobre Airbnb")
        st.markdown("""
        - Plataforma líder de alojamiento
        - Más de 7 millones de listados
        - Opciones para todos los presupuestos
        - Experiencias locales únicas
        """)

    st.markdown("---")
    st.subheader("📊 Objetivo del Dashboard")
    st.markdown("""
    Este dashboard analiza datos de Airbnb en Dallas para ayudar a:
    - Entender el mercado de alquileres temporales
    - Identificar tendencias y patrones
    - Predecir precios y características clave
    """)

    # Imagen de Airbnb
    st.image(r"C:\Users\leirb\Downloads\puebas\img\airbnb.png", 
             use_container_width=True,  # Cambiado a use_container_width
             caption="Plataforma Airbnb - Alojamientos en Dallas")

@st.cache_resource
def load_data():
    df = pd.read_csv("Dallas_limpio.csv")

    # Eliminar columna Unnamed si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Manejar la columna host_is_superhost
    if "host_is_superhost" in df.columns:
        # Primero verificar el tipo actual de datos
        if df["host_is_superhost"].dtype == object:
            # Si ya es string (f/t), crear versión numérica
            df["host_is_superhost_num"] = df["host_is_superhost"].map({'f': 0, 't': 1})
            # Opcional: convertir a texto más descriptivo
            df["host_is_superhost"] = df["host_is_superhost"].map({'f': "No", 't': "Sí"})
        else:
            # Si es numérico (0/1), crear versión categórica
            df["host_is_superhost_num"] = df["host_is_superhost"].astype(int)
            df["host_is_superhost"] = df["host_is_superhost"].map({0: "No", 1: "Sí"})

    # Resto del código permanece igual
    if "price" in df.columns:
        avg_price = df["price"].mean()
        df["high_price"] = (df["price"] > avg_price).astype(int)

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    numeric_cols = numeric_df.columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    unique_room_types = df['room_type'].unique() if 'room_type' in df.columns else []

    binary_cols = []
    for col in numeric_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and (set(unique_vals) == {0, 1} or set(unique_vals) == {False, True}):
            binary_cols.append(col)

    return df, numeric_cols, text_cols, unique_room_types, numeric_df, binary_cols

df, numeric_cols, text_cols, unique_room_types, numeric_df, binary_cols = load_data()

# Función para crear el mapa
def create_map(data):
    # Crear mapa centrado en Dallas
    m = folium.Map(location=[32.7767, -96.7970], zoom_start=12, tiles='CartoDB dark_matter')

    # Añadir marcadores para cada propiedad
    for idx, row in data.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            # Crear popup con información
            popup_text = f"""
            <b>Propiedad #{idx}</b><br>
            <b>Tipo:</b> {row.get('property_type', 'N/A')}<br>
            <b>Habitaciones:</b> {row.get('bedrooms', 'N/A')}<br>
            <b>Precio:</b> ${row.get('price', 'N/A')}<br>
            <b>Anfitrión:</b> {row.get('host_name', 'N/A')}<br>
            <b>Barrio:</b> {row.get('neighbourhood_cleansed', 'N/A')}
            """

            # Color del marcador según el precio
            price = row.get('price', 0)
            if price > data['price'].quantile(0.75):
                color = 'red'
            elif price > data['price'].median():
                color = 'orange'
            else:
                color = 'green'

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"${price} - {row.get('property_type', '')}",
                icon=folium.Icon(color=color, icon='home')
            ).add_to(m)

    return m

# Sidebar
st.sidebar.title("Dashboard de Dallas")
st.sidebar.markdown("---")
view_option = st.sidebar.selectbox("Selecciona vista", [
    "Inicio",  # Nueva opción de inicio
    "Vista 1: Gráfico de Líneas", 
    "Vista 2: Diagrama de Dispersión", 
    "Vista 3: Gráfico Circular", 
    "Vista 4: Otros Gráficos",
    "Vista 5: Regresión Lineal Simple",
    "Vista 6: Regresión Lineal Múltiple",
    "Vista 7: Regresión Logística",
    "Vista 8: Mapa de Propiedades"  # Nueva opción para el mapa
], key='view_select')

st.sidebar.markdown("---")
mostrar_dataset = st.sidebar.checkbox("Mostrar Dataset", key='show_data')
mostrar_columnas_string = st.sidebar.checkbox("Mostrar columnas tipo texto", key='show_text_cols')

# Mostrar vista seleccionada
if view_option == "Inicio":
    show_welcome()
else:
    if mostrar_dataset:
        st.subheader("Dataset completo")
        with st.expander("Ver datos completos"):
            st.write(df)
            st.write("Columnas:", df.columns)
            st.write("Estadísticas descriptivas:", df.describe())

    if mostrar_columnas_string:
        st.subheader("Columnas tipo texto (STRING)")
        st.write(text_cols)

    # Vista 1: Gráfico de Líneas
    if view_option == "Vista 1: Gráfico de Líneas":
        st.title("📈 DALLAS - Tendencias por Tipo de Habitación")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            variables_lineplot = st.multiselect("Variables numéricas", options=numeric_cols, key='line_vars')
        with col2:
            categoria_lineplot = st.selectbox("Tipo de Habitación", options=unique_room_types, key='room_type')

        if variables_lineplot:
            data = df[df['room_type'] == categoria_lineplot]
            if not data.empty:
                data_features = data[variables_lineplot]
                figure1 = px.line(
                    data_frame=data_features,
                    x=data_features.index,
                    y=variables_lineplot,
                    title='Tendencias por Tipo de Habitación',
                    width=1600, 
                    height=600,
                    color_discrete_sequence=["#ff4b4b"],
                    template="plotly_dark"
                )
                st.plotly_chart(figure1, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para graficar")
        else:
            st.warning("Selecciona al menos una variable para graficar")

    # Vista 2: Diagrama de Dispersión
    elif view_option == "Vista 2: Diagrama de Dispersión":
        st.title("🖇️ DALLAS - Diagrama de Dispersión")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            x_selected = st.selectbox("Eje X", options=numeric_cols, key='scatter_x')
        with col2:
            y_selected = st.selectbox("Eje Y", options=numeric_cols, key='scatter_y')

        figure2 = px.scatter(
            data_frame=df, 
            x=x_selected, 
            y=y_selected,
            title='Relación entre variables',
            color_discrete_sequence=["#ff4b4b"],
            template="plotly_dark"
        )
        st.plotly_chart(figure2, use_container_width=True)

    # Vista 3: Gráfico Circular
    elif view_option == "Vista 3: Gráfico Circular":
        st.title("🍕 DALLAS - Gráfico Circular")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            var_cat = st.selectbox("Variable Categórica", options=text_cols, key='pie_cat')
        with col2:
            var_num = st.selectbox("Variable Numérica", options=numeric_cols, key='pie_num')

        try:
            figure3 = px.pie(
                data_frame=df, 
                names=var_cat, 
                values=var_num,
                title='Distribución por Categoría',
                width=1600, 
                height=600,
                color_discrete_sequence=px.colors.sequential.Reds_r,
                template="plotly_dark"
            )
            st.plotly_chart(figure3, use_container_width=True)
        except Exception as e:
            st.error(f"No se puede graficar esta combinación. Error: {str(e)}")

    # Vista 4: Otros gráficos
    elif view_option == "Vista 4: Otros Gráficos":
        st.title("📊 DALLAS - Gráficos Adicionales")
        st.markdown("---")

        st.subheader("Histograma de Precios")
        if "price" in df.columns:
            fig_hist = px.histogram(
                df, 
                x="price", 
                nbins=50, 
                title="Distribución de precios",
                color_discrete_sequence=["#ff4b4b"],
                template="plotly_dark"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No se encuentra la columna 'price'")

        st.subheader("Comparativa de Superhosts")
        if "host_is_superhost_num" in df.columns and "price" in df.columns:
            fig_bar = px.bar(
                df.groupby("host_is_superhost")["price"].mean().reset_index(),
                x="host_is_superhost", 
                y="price", 
                title="Precio promedio por Superhost",
                color="host_is_superhost",
                color_discrete_map={"No": "#ff4b4b", "Sí": "#4bff8f"},
                labels={"host_is_superhost": "Es Superhost"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No se encuentran las columnas necesarias ('host_is_superhost' o 'price')")

    # Vista 5: Regresión Lineal Simple
    elif view_option == "Vista 5: Regresión Lineal Simple":
        st.title("📉 DALLAS - Regresión Lineal Simple")
        st.markdown("---")

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                X_col = st.selectbox("Variable Independiente (X)", options=numeric_cols, key='linreg_x')
            with col2:
                y_col = st.selectbox("Variable Dependiente (y)", options=numeric_cols, key='linreg_y')

            temp_df = df[[X_col, y_col]].dropna()
            X = temp_df[[X_col]].values
            y = temp_df[y_col].values

            if len(X) > 0 and len(y) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Coeficientes
                st.subheader("Coeficientes del Modelo")
                coef_df = pd.DataFrame({
                    'Componente': ['Coeficiente', 'Intercepto'],
                    'Valor': [model.coef_[0], model.intercept_]
                })
                st.dataframe(coef_df, hide_index=True)

                # Crear DataFrame para plotly
                plot_df = pd.DataFrame({
                    X_col: X_test.flatten(),
                    y_col: y_test,
                    'Predicciones': y_pred
                })

                # Gráfico con línea de regresión y predicciones
                fig = px.scatter(
                    plot_df, 
                    x=X_col, 
                    y=y_col,
                    title=f"Regresión Lineal Simple: {X_col} vs {y_col}",
                    labels={X_col: X_col, y_col: y_col},
                    color_discrete_sequence=["#ff4b4b"],
                    template="plotly_dark"
                )

                # Línea de regresión (naranja)
                fig.add_scatter(
                    x=plot_df[X_col], 
                    y=plot_df['Predicciones'], 
                    mode='lines', 
                    name='Línea de Regresión',
                    line=dict(color="#ff9a4b", width=3)
                )

                # Predicciones individuales (verde)
                fig.add_scatter(
                    x=plot_df[X_col],
                    y=plot_df['Predicciones'],
                    mode='markers',
                    name='Predicciones',
                    marker=dict(color='#4bff8f', size=8, line=dict(width=1, color='DarkSlateGrey'))
                )

                st.plotly_chart(fig, use_container_width=True)

                # Mapa de calor de correlación
                st.subheader("Mapa de Calor de Correlación")
                corr_matrix = df[[X_col, y_col]].corr()

                fig_heat, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap="Oranges", ax=ax)
                ax.set_title("Correlación entre Variables")
                st.pyplot(fig_heat)

                # Tabla de predicciones
                st.subheader("Predicciones vs Valores Reales")
                pred_df = pd.DataFrame({
                    'Real': y_test[:20],
                    'Predicción': y_pred[:20],
                    'Diferencia': abs(y_test[:20] - y_pred[:20])
                })
                st.dataframe(pred_df.style.format("{:.2f}"))

    # Vista 6: Regresión Lineal Múltiple
    elif view_option == "Vista 6: Regresión Lineal Múltiple":
        st.title("📊 DALLAS - Regresión Lineal Múltiple")
        st.markdown("---")

        if len(numeric_cols) >= 2:
            X_cols = st.multiselect("Variables Independientes (X)", options=numeric_cols, key='multireg_x')
            y_col = st.selectbox("Variable Dependiente (y)", options=numeric_cols, key='multireg_y')

            if len(X_cols) >= 1:
                temp_df = df[X_cols + [y_col]].dropna()
                X = temp_df[X_cols].values
                y = temp_df[y_col].values

                if len(X) > 0 and len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Solo coeficientes
                    st.subheader("Coeficientes del Modelo")
                    coef_df = pd.DataFrame({
                        'Variable': X_cols + ['Intercepto'],
                        'Coeficiente': list(model.coef_) + [model.intercept_]
                    })
                    st.dataframe(coef_df, hide_index=True)

                    # Crear DataFrame para plotly
                    plot_df = pd.DataFrame({
                        'Real': y_test,
                        'Predicciones': y_pred
                    })

                    # Gráfico mejorado con líneas diferenciadas
                    fig = px.scatter(
                        plot_df, 
                        x='Real', 
                        y='Predicciones',
                        title="Valores Reales vs Predicciones",
                        labels={'Real': 'Valores Reales', 'Predicciones': 'Predicciones'},
                        color_discrete_sequence=["#ff4b4b"],
                        template="plotly_dark"
                    )

                    # Línea de regresión ideal (naranja)
                    min_val = min(min(y_test), min(y_pred))
                    max_val = max(max(y_test), max(y_pred))
                    fig.add_shape(
                        type="line", 
                        x0=min_val, 
                        y0=min_val,
                        x1=max_val, 
                        y1=max_val,
                        line=dict(color="#ff9a4b", dash="dash", width=2),
                        name="Regresión Ideal"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Mapa de calor de correlación entre variables
                    st.subheader("Mapa de Calor de Correlación")
                    corr_matrix = temp_df.corr()

                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Oranges',
                        title="Correlación entre Variables",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # Tabla de predicciones
                    st.subheader("Predicciones vs Valores Reales")
                    pred_df = pd.DataFrame({
                        'Real': y_test[:20],
                        'Predicción': y_pred[:20],
                        'Diferencia': abs(y_test[:20] - y_pred[:20])
                    })
                    st.dataframe(pred_df.style.format("{:.2f}"))
       # Vista 7: Regresión Logística 
    elif view_option == "Vista 7: Regresión Logística":
        st.title("🔮 DALLAS - Regresión Logística")
        st.markdown("---")

        if len(numeric_cols) >= 1 and len(binary_cols) >= 1:
            st.write("**Variables disponibles para clasificación:**", binary_cols)

            X_cols = st.multiselect(
                "Variables Independientes (X)", 
                options=[col for col in numeric_cols if col not in binary_cols],
                key='logreg_x'
            )
            y_col = st.selectbox(
                "Variable Dependiente (y - binaria)", 
                options=binary_cols,
                key='logreg_y'
            )

            if len(X_cols) >= 1:
                temp_df = df[X_cols + [y_col]].dropna()
                X = temp_df[X_cols].values
                y = temp_df[y_col].values

                if len(X) > 0 and len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para la curva ROC

                    # Gráfico de coeficientes
                    fig = px.bar(
                        x=X_cols, 
                        y=model.coef_[0], 
                        title="Importancia de Variables",
                        labels={'x': 'Variables', 'y': 'Coeficientes'},
                        color_discrete_sequence=["#ff9a4b"],
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Matriz de confusión con estilo de mapa de calor
                    st.subheader("Matriz de Confusión")
                    conf_matrix = pd.crosstab(
                        pd.Series(y_test, name='Real'), 
                        pd.Series(y_pred, name='Predicción'),
                        rownames=['Real'],
                        colnames=['Predicción']
                    )

                    # Crear matriz de confusión visual con Plotly
                    fig_conf = px.imshow(
                        conf_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Reds',
                        title="Matriz de Confusión",
                        labels=dict(x="Predicción", y="Real", color="Cantidad"),
                        template="plotly_dark"
                    )

                    # Personalizar hover data
                    fig_conf.update_traces(
                        hovertemplate="<b>Real</b>: %{y}<br><b>Predicción</b>: %{x}<br><b>Cantidad</b>: %{z}<extra></extra>"
                    )

                    # Añadir anotaciones con estilo
                    fig_conf.update_layout(
                        xaxis=dict(tickmode='array', tickvals=[0, 1]),
                        yaxis=dict(tickmode='array', tickvals=[0, 1]),
                        coloraxis_colorbar=dict(title="Casos")
                    )

                    st.plotly_chart(fig_conf, use_container_width=True)

                    # Métricas de rendimiento
                    accuracy = accuracy_score(y_test, y_pred)
                    st.subheader("Métricas de Rendimiento")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Exactitud", f"{accuracy:.2%}", 
                                 help="Porcentaje de predicciones correctas")
                    with col2:
                        tn, fp, fn, tp = conf_matrix.values.ravel()
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        st.metric("Precisión", f"{precision:.2%}", 
                                 help="Verdaderos positivos / Predicciones positivas")
                    with col3:
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        st.metric("Sensibilidad", f"{recall:.2%}", 
                                 help="Verdaderos positivos / Reales positivos")

    # Vista 8: Mapa de Propiedades
    elif view_option == "Vista 8: Mapa de Propiedades":
        st.title("🗺️ Mapa de Propiedades en Dallas")
        st.markdown("---")

        st.write("""
        Este mapa muestra la ubicación de todas las propiedades de Airbnb en Dallas.
        Cada marcador representa una propiedad y su color indica el rango de precio:
        - 🔴 Rojo: Precios altos 
        - 🟠 Naranja: Precios medios
        - 🟢 Verde: Precios bajos
        """)

        # Filtros para el mapa
        st.subheader("Filtros")
        col1, col2 = st.columns(2)

        with col1:
            min_price = st.number_input("Precio mínimo", min_value=0, value=0, step=10)
            room_type_filter = st.selectbox(
                "Tipo de habitación", 
                options=["Todos"] + list(unique_room_types)
            )

        with col2:
            max_price = st.number_input(
                "Precio máximo", 
                min_value=0, 
                value=int(df['price'].max()) if 'price' in df.columns else 1000,
                step=10
            )

        # Aplicar filtros
        filtered_data = df.copy()
        if 'price' in df.columns:
            filtered_data = filtered_data[
                (filtered_data['price'] >= min_price) & 
                (filtered_data['price'] <= max_price)
            ]

        if room_type_filter != "Todos":
            filtered_data = filtered_data[filtered_data['room_type'] == room_type_filter]

        # Mostrar estadísticas de los filtros
        st.write(f"Mostrando {len(filtered_data)} propiedades de {len(df)} totales")

        # Crear y mostrar el mapa
        st.subheader("Mapa Interactivo")
        with st.spinner("Generando mapa..."):
            m = create_map(filtered_data)
            folium_static(m, width=1200, height=700)

        # Mostrar algunas propiedades destacadas
        st.subheader("Propiedades Destacadas")
        if 'price' in df.columns:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Propiedad más cara", 
                         f"${filtered_data['price'].max():.2f}",
                         help=f"Tipo: {filtered_data.loc[filtered_data['price'].idxmax(), 'property_type']}")
            with cols[1]:
                st.metric("Precio promedio", 
                         f"${filtered_data['price'].mean():.2f}")
            with cols[2]:
                st.metric("Propiedad más barata", 
                         f"${filtered_data['price'].min():.2f}",
                         help=f"Tipo: {filtered_data.loc[filtered_data['price'].idxmin(), 'property_type']}")
