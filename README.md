# OpenQA — Multi-Agent Investment Recommendation System

Sistema de recomendación de inversiones basado en múltiples agentes, construido sobre modelos generativos, herramientas externas y Retrieval-Augmented Generation (RAG).

Este proyecto extiende la práctica original (Reasoning + Tools + RAG) y la convierte en un **sistema multiagente estructurado**, donde cada agente tiene un rol claro dentro del proceso de toma de decisiones.

---

## 🧠 Arquitectura del sistema

El sistema divide el problema en etapas:

```text
Usuario
  ↓
Orchestrator Agent
  ↓
Market Intelligence Agent
  ├─ stock_price
  ├─ company_fundamentals
  ├─ company_events
  ├─ internet_search
  └─ RAG
  ↓
Recommendation Agent (Fin-R1)
  ↓
Critic / Risk Agent
  ↓
Respuesta final
````

### Roles de los agentes

* **Orchestrator Agent**

  * Interpreta la query del usuario
  * Extrae empresa, ticker, perfil de riesgo y horizonte

* **Market Intelligence Agent**

  * Obtiene datos objetivos del mercado
  * Usa tools + RAG
  * Genera un informe estructurado

* **Recommendation Agent (Fin-R1)**

  * Construye una tesis de inversión
  * Usa un modelo especializado en razonamiento financiero

* **Critic / Risk Agent**

  * Verifica que la recomendación esté bien fundamentada
  * Evalúa riesgos
  * Decide si hay suficiente evidencia

---

## 📂 Estructura del proyecto

```text
openqa/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── api/
│   └── app.py
│
└── src/
    ├── main.py
    │
    ├── agents/
    │   ├── orchestrator_agent.py
    │   ├── market_intelligence_agent.py
    │   ├── recommendation_agent.py
    │   ├── critic_risk_agent.py
    │   └── investment_multiagent_system.py
    │
    ├── models/
    │   ├── general_model.py
    │   └── fin_model.py
    │
    ├── rlm/
    ├── tool_use/
    ├── rag/
    └── react/   # legacy
```

---

## ⚙️ Modelos utilizados

El sistema utiliza **dos modelos distintos**:

* **Modelo general**

  * Orchestrator + Critic
  * Basado en el modelo entrenado en la práctica

* **Modelo financiero (Fin-R1)**

  * Solo para el Recommendation Agent
  * Especializado en análisis financiero

Esto permite separar:

* lógica general
* razonamiento financiero específico

---

## 🔌 API disponible

### Endpoints de la práctica

* `POST /phase1/reasoning`
* `POST /phase2/tools`
* `POST /phase3/rag`

### Endpoint principal del proyecto

* `POST /investment/recommendation`

#### Ejemplo:

```json
{
  "prompt": "Analiza Nvidia y dime si tendría sentido entrar ahora para un inversor moderado a 12 meses."
}
```

---

## ▶️ Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 2. Cargar RAG (muy importante)

```bash
python -m src.rag.load_dataset
```

---

### 3. Ejecutar sistema multiagente (CLI)

```bash
python -m src.main
```

---

### 4. Ejecutar API

```bash
python api/app.py
```

Acceso:

* API: [http://0.0.0.0:8045](http://0.0.0.0:8045)
* Docs: [http://0.0.0.0:8045/docs](http://0.0.0.0:8045/docs)

---

## 🔑 Variables de entorno

Crear `.env`:

```env
TAVILY_API_KEY=xxxx
ALPHAVANTAGE_API_KEY=xxxx
ENABLE_LEGACY_REACT=false
```

---

## ⚠️ Filosofía del sistema

Este sistema **NO da recomendaciones ciegas**.

Si:

* no identifica la empresa
* no hay datos suficientes
* o la recomendación no está bien fundamentada

→ responde de forma prudente.

Esto es clave para el proyecto.

---

## 📊 Qué se puede evaluar

Este sistema permite comparar:

* Single agent vs multiagente
* Con vs sin RAG
* Con vs sin critic agent

Y medir:

* coherencia
* grounding
* cobertura de riesgos
* calidad de la recomendación

---

## 🚀 Estado del proyecto

* Arquitectura multiagente implementada
* Integración con tools y RAG
* Fin-R1 integrado como recommendation agent
* Sistema legacy mantenido solo para comparación

El flujo principal ahora es el sistema multiagente.

---

