import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(
    title="Darukaa Biodiversity Intelligence AI",
    description="Evidence-backed biodiversity recommendation system",
    version="1.0.0"
)


# -------------------------
# Load environmental knowledge
# -------------------------

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "knowledge_base.json", "r", encoding="utf-8") as file:
    KNOWLEDGE_BASE = json.load(file)


# Create searchable documents from the knowledge base
documents = []

for item in KNOWLEDGE_BASE:
    text = " ".join([
        item["topic"],
        " ".join(item["conditions"]),
        item["recommendation"],
        item["reasoning"],
        " ".join(item["metrics"])
    ])
    documents.append(text)


vectorizer = TfidfVectorizer(stop_words="english")
knowledge_vectors = vectorizer.fit_transform(documents)


# -------------------------
# Input schema
# -------------------------

class EnvironmentalInput(BaseModel):
    query: Optional[str] = None
    soil_organic_carbon: Optional[float] = None
    rainfall: Optional[str] = None
    land_use: Optional[str] = None
    region: Optional[str] = None
    temperature_condition: Optional[str] = None
    pollution: Optional[bool] = None
    deforestation: Optional[bool] = None
    species_richness: Optional[str] = None


# -------------------------
# Retrieval
# -------------------------

def retrieve_knowledge(text: str, top_k: int = 3):

    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        knowledge_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        if similarities[index] > 0:
            item = KNOWLEDGE_BASE[index].copy()
            item["retrieval_score"] = round(
                float(similarities[index]), 3
            )
            results.append(item)

    return results


# -------------------------
# Multi-metric reasoning
# -------------------------

def build_environmental_context(data: EnvironmentalInput):

    context = []

    if data.query:
        context.append(data.query)

    if data.soil_organic_carbon is not None:
        context.append(
            f"soil organic carbon {data.soil_organic_carbon}%"
        )

        if data.soil_organic_carbon < 1:
            context.append("low organic carbon degraded soil")

    if data.rainfall:
        context.append(f"{data.rainfall} rainfall")

    if data.land_use:
        context.append(data.land_use)

    if data.region:
        context.append(data.region)

    if data.temperature_condition:
        context.append(data.temperature_condition)

    if data.pollution:
        context.append("pollution water contamination")

    if data.deforestation:
        context.append("deforestation fragmented habitat")

    if data.species_richness:
        context.append(f"{data.species_richness} species richness")

    return " ".join(context)


def missing_information(data: EnvironmentalInput):

    missing = []

    if data.soil_organic_carbon is None:
        missing.append("soil organic carbon percentage")

    if not data.rainfall:
        missing.append("rainfall condition")

    if not data.land_use:
        missing.append("land-use type")

    return missing


# -------------------------
# API
# -------------------------

@app.get("/")
def home():

    return {
        "system": "Darukaa Biodiversity Intelligence AI",
        "status": "running",
        "description":
            "RAG-inspired environmental decision-support system",
        "endpoint": "/recommend"
    }


@app.post("/recommend")
def recommend(data: EnvironmentalInput):

    missing = missing_information(data)

    # Conversational clarification
    if len(missing) >= 2:
        return {
            "status": "needs_more_information",
            "message":
                "I need additional environmental information "
                "before making a reliable recommendation.",
            "clarifying_questions": [
                f"Please provide {item}."
                for item in missing
            ]
        }

    context = build_environmental_context(data)

    retrieved = retrieve_knowledge(context)

    if not retrieved:
        return {
            "status": "insufficient_knowledge",
            "message":
                "The current knowledge base does not contain "
                "enough evidence for this environmental scenario."
        }

    recommendations = []

    for item in retrieved:

        recommendations.append({
            "recommendation":
                item["recommendation"],

            "scientific_reasoning":
                item["reasoning"],

            "impacted_metrics":
                item["metrics"],

            "time_horizon":
                item["time_horizon"],

            "evidence_source":
                item["source"],

            "retrieval_confidence":
                item["retrieval_score"]
        })

    return {
        "status": "success",

        "environmental_context":
            context,

        "reasoning_summary":
            "Recommendations were selected by retrieving "
            "environmental knowledge relevant to multiple "
            "conditions supplied by the user.",

        "recommendations":
            recommendations
    }
