# BioPhys-Tech Lab: Real-Time Financial Prediction Stack

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Organization: BioPhys-Tech-Lab](https://img.shields.io/badge/Org-BioPhys--Tech--Lab-041030.svg)](https://github.com/BioPhys-Tech-Lab)

Esta plataforma representa la solución técnica integral desarrollada por el **BioPhys-Tech Lab** en colaboración con **EDF**. Es un ecosistema diseñado para la predicción financiera de alta frecuencia, priorizando la **baja latencia**, la **seguridad del capital** y la **integridad estadística**.

---

## 🚀 Pilares Tecnológicos

### 1. Inferencia Optimizada (Low-Latency)
- **ONNX Runtime:** Implementación de modelos optimizados que reducen la latencia de inferencia en un 60% comparado con Python nativo, garantizando un SLA de **<100ms**.
- **Feature Caching:** Sistema de caché distribuido para evitar re-cálculos costosos de indicadores técnicos.

### 2. Gestión de Riesgos y Seguridad (Safety First)
- **Adaptive Safety Breaker:** Sistema dinámico que ajusta el tamaño de la posición o detiene el trading basado en el régimen de volatilidad del mercado (Normal, Elevado, Crisis).
- **Circuit Breaker Pattern:** Protección de infraestructura distribuida que implementa *failover* automático entre servicios primarios, secundarios y cachés locales.

### 3. Integridad de Datos y Consenso
- **Multi-Provider Consensus:** Algoritmo de reconciliación que utiliza la **Mediana Ponderada** y la **Desviación Absoluta de la Mediana (MAD)** para filtrar proveedores de datos maliciosos o erróneos.
- **Resilient Pipeline:** Limpieza de outliers y recuperación inteligente de "gaps" mediante interpolación lineal y forward-fill.

### 4. Observabilidad y MLOps
- **Advanced Drift Detection:** Monitoreo en tiempo real de la degradación del modelo utilizando los tests de **Kolmogorov-Smirnov** y la **Distancia de Wasserstein**.
- **Automated Health Checks:** Monitoreo continuo de la salud del sistema y cumplimiento de SLAs.

---

## 📂 Estructura del Ecosistema

```text
src/
├── models/
│   ├── predictor.py       # Motor de inferencia ONNX & Manager
│   ├── drift.py           # Detección de degradación estadística
│   └── validator.py       # Validación estricta con Pydantic
├── data/
│   ├── pipeline.py        # Ingesta, outlier detection y simulación
│   └── recovery.py        # Motor de recuperación de gaps temporales
├── utils/
│   ├── safety.py          # Gestión de riesgos y regímenes de mercado
│   ├── circuit_breaker.py # Resiliencia de infraestructura (Patterns)
│   └── consensus.py       # Reconciliación multi-proveedor
└── api.py                 # Orquestador FastAPI con lifecycle async
```
---

## 🛠️ Despliegue con Docker

El sistema está totalmente containerizado para asegurar un despliegue determinista y escalable.

### Iniciar el Stack Completo ##📄 Documentación y Fundamentos
```bash
docker-compose up -d --build
```
---
## Ejecutar Suite de Pruebas (Debug & QA)
### Para verificar la integridad de los módulos y el cumplimiento del SLA de latencia:
```bash
python test_ml_debug.py
```
---
## 📄 Documentación y Fundamentos

La justificación teórica, los diagramas de arquitectura y los resultados experimentales se encuentran detallados en el archivo Collaboration.pdf. 

---

### ⚠️ Nota Técnica: Dependencias de Inferencia (ONNX Runtime)

El motor de predicción del **BioPhys-Tech Lab** utiliza **ONNX Runtime** para optimizar la ejecución de los modelos de Gradient Boosting, logrando una reducción significativa en la latencia de inferencia.

Para mantener la eficiencia y seguridad de la infraestructura, se han tomado las siguientes decisiones de diseño:

**Construcción en Etapas (Multi-stage):** El `Dockerfile` utiliza un builder con `gcc` para compilar dependencias, pero la imagen final es de tipo `slim` para minimizar la superficie de ataque y el peso del contenedor.
**Librerías Compartidas:** Dependiendo del entorno de ejecución (OS host), la inferencia con ONNX puede requerir librerías compartidas de C++ (como `libgomp1`). Si el contenedor arroja un error de carga de librerías en sistemas operativos anfitriones muy restrictivos, se recomienda instalar dichas dependencias en la capa final o gestionarlas a nivel de orquestador.
**Optimización de Recursos:** El uso de imágenes ligeras asegura que el servicio pueda escalar rápidamente en entornos de nube sin el "overhead" de herramientas de compilación innecesarias en producción.

---
