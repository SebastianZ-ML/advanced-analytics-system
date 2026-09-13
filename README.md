# Plataforma Multiagente de Analítica Avanzada y Adaptativa (PoC)

Prueba de concepto funcional de arquitectura multiagente orientada a la **analítica reproducible, defendible y auditable de datos de negocio**.

Convierte preguntas en lenguaje natural y tablas dispersas en diagnósticos causales defendibles, eliminando cifras alucinadas mediante contratos tipados de datos, validación determinista independiente y trazabilidad de procedencia de extremo a extremo.

---

## 1. Arquitectura y Flujo de los 10 Agentes

El sistema desacopla estrictamente el razonamiento y la orquestación de la ejecución matemática y la validación. Cada agente produce y consume artefactos tipados con esquemas **Pydantic v2**:

```
[Usuario / Pregunta de Negocio] 
       │
       ▼
[Agente A: Organizador] ──(ObjectiveSpec)──► Identifica decisión, métrica primaria, períodos y ambigüedades.
       │
       ▼
[Agente B: Auditor de Datos] ──(DataCatalog & DataQualityReport)──► Perfila tipos, nulos, ceros iniciales y huérfanos.
       │
       ▼
[Agente C: Metodólogo] ──(AnalysisPlan)──► Selecciona métodos registrados del catálogo con respaldo bibliográfico.
       │
       ▼
[Agente D: Preparador] ──(TransformationRecords)──► Deduplicación, limpieza de fechas y uniones con factor 1.0x.
       │
       ▼
[Agente E: Ejecutor Analítico] ──(AnalysisResults)──► Resumen, series temporales, cohortes y descomposiciones.
       │
       ▼
[Agente F: Validador Analítico] ──(ValidationReport)──► Control determinista: reconciliación 100%, finitud y auto-reparación.
       │
       ▼
[Agente G: Intérprete] ──(InsightReport)──► Separa hechos probados (result_id) de hipótesis y limitaciones.
       │
       ▼
[Agente H: Constructor Dashboard] ──(DashboardSpec)──► Ensambla tarjetas y gráficos a partir de componentes aprobados.
       │
       ▼
[Agente I: Validador Dashboard] ──► Comprueba que no existan resultados rechazados y alinea períodos.
       │
       ▼
[Agente J: Asistente Conversacional] ──(ChatAnswer)──► Responde con citas de procedencia y recálculos con filtros reales.
```

---

## 2. Principios no negociables implementados

- **Inmutabilidad**: Los archivos originales (`orders.csv`, `customers.csv`, `products.xlsx`) se leen y conservan intactos. Se calcula y almacena su hash SHA-256.
- **Trazabilidad estricta**: Cada cifra del dashboard tiene una ruta verificable:
  `Archivo crudo ➔ Tabla ➔ Transformación ➔ Operación analítica ➔ Validación matemática ➔ Visualización`.
- **Cero alucinación cuantitativa**: Ningún LLM ni agente genera cifras numéricas de forma libre. Todo cálculo se ejecuta en el motor DuckDB/pandas.
- **Reconciliación matemática al 100%**: Las descomposiciones por dimensiones (waterfall) demuestran que `sum(delta_dimensiones) == delta_total` dentro de una tolerancia estricta (< 0.01).
- **Control de cardinalidad en uniones**: Antes de unir hechos y dimensiones, se auditan claves duplicadas y se deduplica la dimensión para garantizar un factor de multiplicación exactamente igual a `1.0x`.
- **Preservación de identificadores con ceros a la izquierda**: Columnas como `customer_id` ('00101') se auditan y preservan como texto para evitar pérdidas de claves o truncamientos.
- **Manejo explícito de períodos incompletos**: El sistema detecta meses parciales (ej. 4 días en junio) y genera advertencias explícitas para acotar la comparación a períodos cerrados completos.
- **Separación de hechos e hipótesis**: Las recomendaciones de acción se etiquetan con `is_action_proposal_only=True` y se distinguen de los hallazgos observados.
- **Modo "Demostración sin LLM" transparente**: Funciona de forma completa, defendible y determinista sin requerir API keys de LLM, indicándolo claramente en la interfaz.

---

## 3. Conjunto de Datos Sintéticos de Demostración

Ubicado en `data/demo/`, generado con semilla fija reproducible (`seed=42`):

1. `orders.csv` (2,607 transacciones):
   - Facturación neta y bruta, unidades, precios y descuentos.
   - Variación mensual con una caída deliberada concentrada en el canal **Mayorista / B2B** en abril y mayo de 2026.
   - Fechas inválidas inyectadas intencionalmente (`2026-02-30`, `CORRUPT_DATE`).
   - Pedidos huérfanos con clientes o productos inexistentes.
   - Último mes incompleto (junio 2026, sólo 4 días).
2. `customers.csv` (81 clientes):
   - Identificadores con ceros a la izquierda (`00001` a `00080`).
   - Clave duplicada intencional para verificar deduplicación (`00142`).
3. `products.xlsx`:
   - Hoja `Products`: Catálogo de productos y precios base.
   - Hoja `Categories`: Jerarquía de departamentos y categorías.
4. `campaigns.csv`:
   - Campañas de marketing con presupuestos y vigencias.
5. `ground_truth.json`:
   - Metadatos contables exactos para verificar la precisión del análisis.

---

## 4. Requisitos y Configuración

### Requisitos previos:
- Python 3.10 o superior (probado en Python 3.12).
- Navegador web moderno (Edge, Chrome, Firefox).

### Instalación de dependencias:
```bash
pip install -r backend/requirements.txt
```

### Configuración de Gemini (Opcional):
La aplicación soporta dos modos de operación:
1. **`LLM_ENABLED`**: Utiliza Google Gemini (`gemini-3.5-flash`) mediante la API oficial de Google GenAI para estructuración semántica, propuesta de planes e interpretaciones fundamentadas.
2. **`DEMO_WITHOUT_LLM`**: Modo determinista por catálogo que funciona sin ninguna clave externa y se identifica de forma visible como "Demostración sin LLM".

Para activar el modo LLM, copia el archivo de ejemplo y agrega tu clave de API:
```bash
cp .env.example .env
```
Edita `.env`:
```env
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-3.5-flash
GEMINI_TIMEOUT_SECONDS=60
GEMINI_MAX_RETRIES=2
```
*Si `GEMINI_API_KEY` se deja vacía o no existe el archivo `.env`, la aplicación se iniciará de forma automática y segura en modo `DEMO_WITHOUT_LLM`.*

---

## 5. Instrucciones de Ejecución

### 1. Iniciar el servidor Backend + Frontend:
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

### 2. Abrir la aplicación:
Abre tu navegador en:
```
http://127.0.0.1:8000
```

### 3. Recorrido de Demostración:
1. En la pestaña **1. Proyectos**, haz clic en **"📥 Cargar Tablas Sintéticas de Demostración"**.
2. Haz clic en **"🚀 Ejecutar Recorrido Vertical Completo"**.
3. Observa la píldora de estado en la cabecera:
   - `🟢 Gemini habilitado (gemini-3.5-flash)` o `🔵 Demostración sin LLM (Motor analítico determinista)`.
4. Explora las secciones:
   - **2. Tablas y Catálogo**: Revisa los perfiles de columnas y la preservación de ceros iniciales.
   - **3. Objetivo y Pregunta**: Observa el distintivo de asistencia de Gemini o reglas deterministas.
   - **4. Calidad y Relaciones**: Revisa las alertas de período incompleto en junio y los registros huérfanos gestionados.
   - **5. Plan Analítico**: Verifica la validación de métodos contra el catálogo determinista.
   - **6. Progreso y Auditoría**: Observa las transiciones de la máquina de estados y transformaciones.
   - **7. Dashboard Validado**: Revisa las tarjetas ejecutivas, gráficos waterfall reconciliados al 100% y el distintivo de cálculo por herramienta determinista (DuckDB/Pandas).
   - **8. Asistente Contextual**: Consulta al chatbot con citas auditadas, recálculos con filtros en caliente y límites explícitos para datos ausentes.
   - **9. Procedencia y Trazabilidad**: Inspecciona la cadena auditable paso a paso.

---

## 6. Baterías de Pruebas Automatizadas (31 de 31 Tests Pasando)

El proyecto cuenta con dos suites de pruebas automatizadas con `pytest`:

### Suite A: Pruebas de Integración con Gemini y Gobernanza (`backend/tests/test_gemini_integration.py`)
16 pruebas que verifican el desacoplamiento, la seguridad de credenciales y los guardianes deterministas:
- Inicio limpio sin clave API y operación determinista de `NoLLMProvider`.
- Validación de contratos tipados Pydantic (`ObjectiveSpec`, `AnalysisPlan`, `InsightReport`).
- Guardián del Metodólogo: rechazo estricto de operaciones o columnas no existentes.
- Guardián del Intérprete: descarte de hallazgos que citen resultados no aprobados.
- Asistente conversacional con citas auditadas y detección inmediata de preguntas no respondibles.
- Redacción obligatoria de secretos en logs y excepciones (`[REDACTED_API_KEY]`).
- Registro de auditoría en SQLite sin credenciales.
- Resistencia ante prompt injection y reintentos con backoff ante errores 429/503.
- Degradación elegante a modo determinista ante fallos de red.
- Preservación exacta al centavo de las cifras calculadas por DuckDB/Pandas.
- Llamada real a la API de Gemini (`gemini-3.5-flash`) con salida estructurada.

```bash
python -m pytest backend/tests/test_gemini_integration.py -v
```

### Suite B: Pruebas de Aceptación Funcional (`backend/tests/test_acceptance.py`)
15 pruebas sobre los criterios no negociables de la prueba de concepto:

| # | Prueba de Aceptación | Resultado |
|---|---|---|
| 1 | Importación de múltiples CSVs y hojas de Excel | **PASSED** |
| 2 | Conservación de identificadores con ceros iniciales | **PASSED** |
| 3 | Detección y prevención de unión que multiplica ventas | **PASSED** |
| 4 | Detección de registros huérfanos | **PASSED** |
| 5 | Tratamiento explícito de fechas inválidas | **PASSED** |
| 6 | Advertencia sobre período incompleto (junio 4 días) | **PASSED** |
| 7 | Reconciliación entre variación total y contribuciones | **PASSED** |
| 8 | Rechazo determinista de valores NaN o infinitos | **PASSED** |
| 9 | Propagación de resultado rechazado hasta dashboard y chat | **PASSED** |
| 10 | Reejecución idéntica con versionado de run_id | **PASSED** |
| 11 | Filtros que actualizan cifras coherentemente | **PASSED** |
| 12 | Pregunta del chatbot que provoca un cálculo real | **PASSED** |
| 13 | Pregunta sin datos recibe limitación explícita | **PASSED** |
| 14 | Manejo de archivos vacíos o malformados | **PASSED** |
| 15 | Cancelación o fallo sin dejar estado engañoso | **PASSED** |

```bash
python -m pytest backend/tests/test_acceptance.py -v
```

---

## 7. Lista Honesta de Capacidades

### Implementadas de extremo a extremo:
- Ingesta y auditoría de CSVs y archivos Excel `.xlsx` multi-hoja.
- Detección de tipos, nulos, ceros a la izquierda, fechas corruptas, períodos incompletos y registros huérfanos.
- Prevención determinista de inflación de uniones con comprobación de factor 1.0x.
- Descomposición aditiva dimensional de caída de ventas con reconciliación al 100%.
- Dinámica de clientes nuevos vs recurrentes con definición explícita de cohorte.
- Validador analítico independiente con ciclo de auto-reparación (hasta 2 intentos).
- Validador de dashboard que bloquea visualización de resultados rechazados.
- Asistente conversacional con 4 tipos de consultas (explicación con procedencia, recálculo con filtros en caliente, detección de nuevo análisis, y declaración explícita de datos no disponibles).
- Persistencia en SQLite de proyectos, archivos, ejecuciones, eventos de auditoría y artefactos tipados.
- UI completa en español con React, TypeScript y Tailwind.

### Capacidades pendientes para siguientes versiones:
- **Conectores SQL directos**: Interfaz base preparada, pero conexiones reales a PostgreSQL y Microsoft SQL Server quedan para el siguiente ciclo.
- **Modelos causales multivariables**: Excluidos deliberadamente en esta fase para no simular inferencia causal no defendible.
- **Pronóstico de series temporales avanzado**: Auditoría de elegibilidad implementada; modelado con Prophet/ARIMA deshabilitado cuando el historial es menor a 12 períodos completos.
- **Autenticación empresarial y control de acceso basado en roles (RBAC)**.
