import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Configuración de página
st.set_page_config(
    page_title="Predicción de Rotación de Empleados",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS personalizado para diseño moderno
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Predicción de rotación de empleados — Employee Attrition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Herramienta analítica para identificar el riesgo de abandono voluntario</div>', unsafe_allow_html=True)

# Cargar preprocesador y modelo de forma segura
@st.cache_resource
def load_assets():
    prep = joblib.load('preprocessor.joblib')
    mod = joblib.load('model.joblib')
    return prep, mod

try:
    preprocessor, model = load_assets()
except Exception as e:
    st.error(f"Error al cargar el modelo o preprocesador: {e}. Por favor, asegúrate de ejecutar primero las celdas de entrenamiento en Colab.")
    st.stop()

tab1, tab2 = st.tabs(["👤 Predicción de un empleado", "📂 Predicción masiva desde Excel"])

with tab1:
    st.subheader("Formulario de datos del empleado")
    st.write("Completa la información del perfil para calcular el riesgo de deserción.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Edad (Años)", min_value=18, max_value=70, value=35, step=1)
        business_travel = st.selectbox("Viajes de Negocio", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        department = st.selectbox("Departamento", ['Research', 'Sales', 'HR'])
        distance = st.number_input("Distancia desde el Hogar (KM)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        education = st.slider("Nivel de Educación (1 a 5)", min_value=1, max_value=5, value=3)
        
    with col2:
        env_satisfaction = st.slider("Satisfacción con Entorno (1 a 4)", min_value=1, max_value=4, value=3)
        gender = st.selectbox("Género", ['Male', 'Female'])
        job_involvement = st.slider("Involucramiento Laboral (1 a 4)", min_value=1, max_value=4, value=3)
        job_level = st.slider("Nivel de Puesto (1 a 5)", min_value=1, max_value=5, value=2)
        job_role = st.selectbox("Rol en la Empresa", ['SalesExec', 'Scientist', 'Technician', 'Manager'])
        
    with col3:
        job_satisfaction = st.slider("Satisfacción de Trabajo (1 a 4)", min_value=1.0, max_value=4.0, value=3.0, step=1.0)
        marital_status = st.selectbox("Estado Civil", ['Married', 'Single', 'Divorced'])
        monthly_income = st.number_input("Ingreso Mensual ($)", min_value=1000.0, max_value=30000.0, value=5500.0, step=100.0)
        overtime = st.selectbox("Trabaja Horas Extra", ['Yes', 'No'])
        total_working_years = st.number_input("Años Totales de Experiencia", min_value=0.0, max_value=50.0, value=8.0, step=0.5)
        years_at_company = st.number_input("Años en la Compañía Actual", min_value=0.0, max_value=40.0, value=5.0, step=0.5)

    if st.button("Predecir Attrition", type="primary"):
        # Crear un dataframe temporal con una fila
        single_employee_df = pd.DataFrame({
            'Age': [age],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'DistanceFromHome': [distance],
            'Education': [education],
            'EnvironmentSatisfaction': [env_satisfaction],
            'Gender': [gender],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MaritalStatus': [marital_status],
            'MonthlyIncome': [monthly_income],
            'OverTime': [overtime],
            'TotalWorkingYears': [total_working_years],
            'YearsAtCompany': [years_at_company]
        })
        
        # Preprocesar datos con el pipeline
        processed_data = preprocessor.transform(single_employee_df)
        
        # Predecir clase y probabilidad
        pred = model.predict(processed_data)[0]
        proba = model.predict_proba(processed_data)[0][1]
        
        # Mostrar resultados con estilo visual impactante
        st.markdown("--- ")
        st.subheader("Resultado del Análisis")
        
        if pred == 1:
            st.error(f"🚨 **Ye — Posible Abandono**: El empleado tiene una alta tendencia a la rotación.")
            st.metric(label="Probabilidad de Attrition", value=f"{proba*100:.2f}%")
        else:
            st.success(f"✅ **No — Probable Permanencia**: El empleado muestra una sólida estabilidad en el puesto.")
            st.metric(label="Probabilidad de Attrition", value=f"{proba*100:.2f}%")
            
        st.caption("⚠️ Nota: Esta predicción corresponde a una estimación estadística basada en patrones históricos de la empresa, no representa una certeza absoluta.")

with tab2:
    st.subheader("Carga masiva de datos de empleados")
    st.write("Sube un archivo en formato Excel (`.xlsx` o `.xls`) que contenga los datos de los empleados para realizar predicciones masivas instantáneas.")
    
    # Ofrecer descarga de la plantilla generada
    with open('plantilla_empleados.xlsx', 'rb') as f:
        st.download_button(
            label="📥 Descargar Plantilla de Ejemplo (Excel)",
            data=f,
            file_name="plantilla_empleados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    uploaded_file = st.file_uploader("Sube tu archivo de Excel aquí", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_excel(uploaded_file)
            st.write(f"Archivo cargado con éxito. Detectadas **{len(input_df)}** filas.")
            
            # Validar que existan todas las columnas requeridas (excepto la objetivo 'Attrition')
            required_cols = [
                'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
                'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
                'OverTime', 'TotalWorkingYears', 'YearsAtCompany'
            ]
            
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if len(missing_cols) > 0:
                st.error(f"❌ No se puede procesar el archivo. Faltan las siguientes columnas obligatorias: {missing_cols}")
            else:
                # Aplicar preprocesador sobre el lote completo de datos
                batch_processed = preprocessor.transform(input_df)
                
                # Obtener predicciones y probabilidades
                predictions = model.predict(batch_processed)
                probabilities = model.predict_proba(batch_processed)[:, 1]
                
                # Agregar resultados conservando las columnas originales intactas
                output_df = input_df.copy()
                output_df['Prediccion_Attrition'] = np.where(predictions == 1, 'Ye', 'No')
                output_df['Probabilidad_Ye'] = np.round(probabilities, 4)
                
                st.write("### Vista Previa de las Predicciones")
                st.dataframe(output_df)
                
                # Permitir descargar el archivo resultante
                towrite = io.BytesIO()
                output_df.to_excel(towrite, index=False, header=True)
                towrite.seek(0)
                st.download_button(
                    label="📥 Descargar Resultados Predictivos en Excel",
                    data=towrite,
                    file_name="predicciones_attrition_completado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
