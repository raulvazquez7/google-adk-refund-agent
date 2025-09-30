## **📋 Scope Final Confirmado**

**Agentes del Sistema:**

1. **CoordinatorAgent** (Router \+ Memory Manager)  
2. **PolicyExpertAgent** (RAG specialist \- ya existe, mejorado)  
3. **TransactionAgent** (Orders \+ Refunds \- ya existe, mejorado)  
4. **ExchangeAgent** 🆕 (Cambios de talla/modelo \+ Stock management)  
5. **ShippingAgent** 🆕 (Tracking de envíos \- mockeado)

### **Funcionalidades:**

* ✅ Refunds (mejorado con paralelización)  
* ✅ Exchanges (cambios con validación de stock)  
* ✅ Shipping tracking (estado de envíos)  
* ✅ Memory Bank (contexto de usuario persistente)  
* ✅ Exception handling robusto

---

## **🎯 FASE 1 \- Paso 1: Entender la Arquitectura Multi-Agente**

Antes de escribir código, necesitas entender **por qué y cómo** funciona esto.

**Concepto Clave: ¿Qué es un Sistema Multi-Agente?**

**Antes (Single Agent):**

User → Agent (hace TODO) → Response

├── RAG

├── Get Order

├── Process Refund

└── Decide qué hacer

**Problema:**

* Un agente generalista no es experto en nada  
* Difícil de mantener (un cambio afecta todo)  
* No escala (no puedes paralelizar tareas especializadas)

---

**Después (Multi-Agent):**

User → Coordinator → Decide qué agente(s) necesita

│

├→ PolicyExpert (solo sabe de políticas)

├→ TransactionAgent (solo sabe de órdenes/refunds)

├→ ExchangeAgent (solo sabe de cambios/stock)

└→ ShippingAgent (solo sabe de envíos)

Coordinator ensambla las respuestas → Response

**Ventajas:**

* ✅ **Especialización:** Cada agente es experto en su dominio  
* ✅ **Paralelización:** Coordinator puede llamar a varios agentes en paralelo  
* ✅ **Mantenibilidad:** Cambios en política no afectan a shipping  
* ✅ **Testeable:** Puedes testear cada agente de forma aislada  
* ✅ **Escalable:** Añadir un nuevo agente no rompe nada

**Concepto Clave: Comunicación Agent-to-Agent (A2A)**

Los agentes necesitan **hablar entre sí**. Hay dos enfoques:

**Enfoque 1: Orquestación Centralizada (Lo que usaremos)\# Coordinator decide quién hace quécoordinator:  if intent \== "refund":    policy \= await policy\_expert.run("check refund policy")    order \= await transaction\_agent.run("get order details")    decision \= coordinator.decide(policy, order)    if approved:      await transaction\_agent.run("process refund")**

**Ventajas:**

* Simple de implementar  
* Control total del flujo  
* Fácil de debuggear

**Desventajas:**

* Coordinator puede volverse complejo  
* Single point of failure

---

**Enfoque 2: Coreografía (Descentralizado)\# Agentes se coordinan solospolicy\_expert → transaction\_agent: "Necesito info de orden X"transaction\_agent → exchange\_agent: "¿Hay stock de talla 26?"exchange\_agent → transaction\_agent: "Sí, hay 5 unidades"**

**Ventajas:**

* Más flexible  
* No hay single point of failure

**Desventajas:**

* Más complejo  
* Difícil de debuggear

**Decisión:** Usaremos **Orquestación** (Enfoque 1\) porque es más didáctico y suficiente para tu caso.

---

**Protocolo de Comunicación A2A**

Los agentes necesitan un "lenguaje común" para hablar. Propongo este protocolo:

**¿Por qué este formato?**

* ✅ Estandarizado: todos los agentes lo entienden  
* ✅ Trazable: trace\_id permite seguir el flujo en Langfuse  
* ✅ Debuggeable: Metadata te dice qué pasó y cuánto costó  
* ✅ Robusto: status \+ error permite manejar fallos

---

## **🛠️ FASE 1 \- Paso 2: Diseñar la Estructura del Código**

Antes de programar, vamos a **diseñar la estructura** del proyecto.

**Estructura Propuesta:**

ReAct Google ADK/

├── src/

│   ├── **init**.py

│   ├── [config.py](http://config.py)                    \# 🆕 Configuración centralizada

│   ├── [protocols.py](http://protocols.py)                 \# 🆕 Definición de AgentRequest/Response

│   ├── memory/                      \# 🆕 Memory Bank

│   │   ├── **init**.py

│   │   └── memory\_manager.py        \# Gestión de memoria de usuario

│   ├── agents/                      \# 🆕 Agentes especializados

│   │   ├── **init**.py

│   │   ├── base\_agent.py            \# Clase base para todos los agentes

│   │   ├── [coordinator.py](http://coordinator.py)           \# CoordinatorAgent

│   │   ├── policy\_expert.py         \# PolicyExpertAgent

│   │   ├── transaction\_agent.py     \# TransactionAgent

│   │   ├── exchange\_agent.py        \# ExchangeAgent

│   │   └── shipping\_agent.py        \# ShippingAgent

│   └── tools/                       \# 🔄 Tools refactorizados

│       ├── **init**.py

│       ├── rag\_tool.py              \# RAG search

│       ├── order\_tool.py            \# Order operations

│       ├── refund\_tool.py           \# Refund processing

│       ├── stock\_tool.py            \# 🆕 Stock management

│       └── shipping\_tool.py         \# 🆕 Shipping tracking (mock)

├── data/

│   ├── company\_refund\_policy\_barefoot.md

│   ├── orders.jsonl

│   ├── inventory.jsonl              \# 🆕 Stock data

│   └── shipping\_tracking.json       \# 🆕 Mock tracking data

├── scripts/

│   ├── 01\_seed\_orders.py

│   ├── 02\_setup\_vector\_search.py

│   └── 03\_seed\_inventory.py         \# 🆕 Cargar stock data

├── [main.py](http://main.py)                          \# CLI interface

└── requirements.txt

---

## **🎓 Conceptos que Aprenderás en Cada Paso**

**Paso 3 (Siguiente): Crear [protocols.py](http://protocols.py) y base\_agent.py**

**Aprenderás:**

* Cómo diseñar protocolos de comunicación entre agentes  
* Herencia en Python para crear agents base  
* Type hints con Pydantic para validación

**Paso 4: Implementar CoordinatorAgent**

**Aprenderás:**

* Intent classification con LLMs  
* Routing basado en intents  
* Orquestación de múltiples agentes  
* Manejo de dependencias entre tasks

**Paso 5: Refactorizar agentes existentes (Policy, Transaction)**

**Aprenderás:**

* Migración de código legacy a nueva arquitectura  
* Separación de concerns  
* Testing de agentes individuales

**Paso 6: Crear ExchangeAgent**

**Aprenderás:**

* Lógica de negocio compleja  
* Transacciones multi-step  
* Rollback en caso de error

**Paso 7: Crear ShippingAgent (mock)**

**Aprenderás:**

* Cómo mockear APIs externas  
* Testing de integraciones

**Paso 8: Implementar MemoryBank**

**Aprenderás:**

* Persistencia de estado en Firestore  
* Context management  
* Privacy considerations

**Paso 9: Paralelización y Exception Handling**

**Aprenderás:**

* asyncio.gather() para paralelización  
* Retries con tenacity  
* Circuit breakers

