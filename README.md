# Darukaa Biodiversity Intelligence AI

An evidence-backed environmental decision-support system that analyzes multiple ecological variables together and retrieves scientifically grounded biodiversity recommendations.

Built for the **Darukaa.Earth AI Biodiversity Intelligence Hackathon**.

---

## Problem Statement

Environmental degradation rarely results from a single variable.

Low soil organic carbon, rainfall stress, monoculture, habitat fragmentation, climate conditions and biodiversity loss can interact with each other.

This project therefore focuses on **multi-metric environmental reasoning** rather than producing generic single-variable recommendations.

---

## Core Capabilities

- Structured environmental knowledge base
- TF-IDF based knowledge retrieval
- Cosine-similarity ranking
- Multi-metric environmental reasoning
- Evidence-backed recommendations
- Scientific source references and URLs
- Clarifying questions when critical information is missing
- Structured JSON input and output
- Confidence/retrieval scores
- Interactive FastAPI documentation

---

## System Architecture

```text
Environmental Input
        |
        v
+-----------------------+
| Input Validation      |
| + Clarification Logic |
+-----------------------+
        |
        v
+-----------------------+
| Environmental Context |
| Builder               |
+-----------------------+
        |
        +----------------------+
        |                      |
        v                      v
+------------------+    +----------------------+
| Multi-Metric     |    | Knowledge Retrieval  |
| Reasoning Engine |    | TF-IDF + Cosine Sim. |
+------------------+    +----------------------+
        |                      |
        +----------+-----------+
                   |
                   v
        +----------------------+
        | Evidence-Backed      |
        | Recommendation Layer |
        +----------------------+
                   |
                   v
        Structured JSON Response
```

---

## Knowledge System

The application uses a structured environmental knowledge base stored in:

`knowledge_base.json`

Each knowledge record contains:

- Environmental conditions
- Recommended intervention
- Scientific reasoning
- Impacted environmental metrics
- Time horizon
- Evidence statement
- Scientific source
- Source URL

Current knowledge areas include:

- Soil organic carbon
- Soil degradation
- Cover crops
- Agroforestry
- Water stress
- Semi-arid agriculture
- Habitat connectivity
- Climate resilience

The current MVP uses **TF-IDF vectorization with cosine similarity** to retrieve the most relevant knowledge records for a given environmental scenario.

---

## Multi-Metric Reasoning

The system does not evaluate environmental variables independently.

For example:

```text
Low Soil Organic Carbon
        +
Low Rainfall
        +
Monoculture
        |
        v
Reduced soil resilience
        +
Water stress
        +
Low habitat diversity
        |
        v
Diversified drought-tolerant agroforestry /
intercropping recommendation
```

The reasoning layer identifies environmental pressures and then explains important cross-variable interactions before retrieving interventions.

---

## Example

### Input

```json
{
  "soil_organic_carbon": 0.3,
  "rainfall": "low",
  "land_use": "monoculture wheat",
  "region": "semi-arid"
}
```

### Example Reasoning

The system identifies that:

- Low soil organic carbon indicates limited organic matter and potentially reduced soil resilience.
- Low rainfall increases water stress.
- Monoculture provides lower structural and habitat diversity than a diversified production system.
- Low soil carbon and water stress interact, making soil-cover and organic-matter restoration important.
- Low-rainfall monoculture creates both climatic and ecological vulnerability.

### Example Recommendation

**Introduce appropriately designed agroforestry together with drought-tolerant crop diversification or intercropping.**

The response also returns:

- Scientific reasoning
- Impacted environmental metrics
- Time horizon
- Evidence statement
- Evidence source
- Source URL
- Retrieval confidence

---

## Scientific Grounding

The knowledge layer currently references authoritative environmental sources including:

### Food and Agriculture Organization of the United Nations (FAO)

Agroforestry:

https://www.fao.org/agroforestry/en

Soil Organic Cover:

https://www.fao.org/conservation-agriculture/in-practice/soil-organic-cover/en/

### Intergovernmental Panel on Climate Change (IPCC)

Special Report on Climate Change and Land:

https://www.ipcc.ch/srccl/

Summary for Policymakers:

https://www.ipcc.ch/srccl/chapter/summary-for-policymakers/

These sources are stored with the relevant knowledge records so retrieved recommendations can expose their evidence provenance.

---

## API

### Health / System Information

```http
GET /
```

### Environmental Recommendation

```http
POST /recommend
```

The recommendation endpoint accepts structured environmental information and returns a knowledge-grounded response.

---

## Conversational Clarification

When insufficient environmental information is supplied, the system avoids immediately generating a recommendation.

Instead, it requests important missing variables such as:

- Soil organic carbon percentage
- Rainfall pattern
- Land-use type

This reduces unsupported recommendations when environmental context is incomplete.

---

## Local Setup

### 1. Clone repository

```bash
git clone https://github.com/ppanchal13/darukaa-biodiversity-ai.git
cd darukaa-biodiversity-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start API

```bash
python -m uvicorn main:app --reload
```

### 4. Open interactive documentation

```text
http://127.0.0.1:8000/docs
```

Use the `POST /recommend` endpoint to test environmental scenarios.

---

## Technology Stack

- Python
- FastAPI
- Pydantic
- Scikit-learn
- TF-IDF
- Cosine Similarity
- JSON structured knowledge base
- Git / GitHub

---

## Database / Schema

For the hackathon MVP, environmental knowledge is stored in a structured JSON knowledge layer.

Conceptual schema:

```text
KnowledgeRecord
├── id
├── topic
├── conditions[]
├── recommendation
├── reasoning
├── metrics[]
├── time_horizon
├── evidence
├── source
└── source_url
```

This keeps the MVP lightweight and reproducible while maintaining explicit evidence provenance.

---

## Current Limitations

This repository represents a hackathon MVP.

Current limitations include:

- Small curated environmental knowledge base
- TF-IDF retrieval rather than semantic embedding retrieval
- No persistent conversational-memory database
- No geospatial analysis layer
- Recommendations require local validation before real-world implementation

These limitations are intentionally documented rather than hidden.

---

## Production Roadmap

A production version could extend the architecture with:

```text
Research Papers / FAO / IPCC / Environmental Datasets
                         |
                         v
                Document Processing
                         |
                         v
                    Embeddings
                         |
                         v
              PostgreSQL + pgvector
                         |
                         v
                 Semantic Retrieval
                         |
                         v
                LLM Reasoning Layer
                         |
            +------------+-------------+
            |                          |
            v                          v
Conversation Memory            Geospatial Context
            |                          |
            +------------+-------------+
                         |
                         v
          Evidence-Backed Recommendation
```

Potential improvements:

- PostgreSQL + pgvector
- Semantic embeddings
- Research-paper ingestion pipeline
- LLM-assisted synthesis
- Persistent conversation memory
- Geo-coordinate and spatial context
- Environmental dataset integrations
- Automated evaluation pipeline
- Docker deployment
- GitHub Actions CI/CD

---

## CI/CD

The current MVP uses GitHub for source control and version history.

A production deployment would add automated testing and deployment using GitHub Actions, containerization and a cloud hosting environment.

---

## Responsible Use

This application is an environmental decision-support prototype.

Recommendations should be adapted using local ecological conditions, field measurements and expert assessment before real-world implementation.

---

## Author

**Parth Panchal**

GitHub: https://github.com/ppanchal13