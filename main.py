import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(
    title="Darukaa Biodiversity Intelligence AI",
    description=(
        "Knowledge-grounded environmental decision-support system "
        "for evidence-backed biodiversity recommendations."
    ),
    version="1.1.0",
)


# ---------------------------------------------------------
# KNOWLEDGE LAYER
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "knowledge_base.json", "r", encoding="utf-8") as file:
    KNOWLEDGE_BASE = json.load(file)


def knowledge_document(item):
    """Convert one structured knowledge record into searchable text."""
    return " ".join(
        [
            item["topic"],
            " ".join(item["conditions"]),
            item["recommendation"],
            item["reasoning"],
            " ".join(item["metrics"]),
            item.get("evidence", ""),
        ]
    )


documents = [knowledge_document(item) for item in KNOWLEDGE_BASE]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
)

knowledge_vectors = vectorizer.fit_transform(documents)


# ---------------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CONTEXT + CLARIFICATION
# ---------------------------------------------------------

def missing_information(data: EnvironmentalInput):
    missing = []

    if data.soil_organic_carbon is None:
        missing.append(
            "What is the approximate soil organic carbon percentage?"
        )

    if not data.rainfall:
        missing.append(
            "How would you describe the rainfall pattern "
            "(low, moderate, high, irregular or seasonal)?"
        )

    if not data.land_use:
        missing.append(
            "What is the current land-use type "
            "(for example monoculture, mixed farming, forest or grassland)?"
        )

    return missing


def build_environmental_context(data: EnvironmentalInput):
    context = []

    if data.query:
        context.append(data.query)

    if data.soil_organic_carbon is not None:
        context.append(
            f"soil organic carbon {data.soil_organic_carbon}%"
        )

        if data.soil_organic_carbon < 1:
            context.extend(
                [
                    "low organic carbon",
                    "degraded soil",
                    "low soil carbon",
                ]
            )

    if data.rainfall:
        context.append(f"{data.rainfall} rainfall")

        if data.rainfall.lower() in ["low", "irregular"]:
            context.append("water stress")

    if data.land_use:
        context.append(data.land_use)

        if "monoculture" in data.land_use.lower():
            context.append("monoculture")

    if data.region:
        context.append(data.region)

    if data.temperature_condition:
        context.append(data.temperature_condition)

        if "high" in data.temperature_condition.lower():
            context.append("climate stress")

    if data.pollution:
        context.extend(
            ["pollution", "environmental stress"]
        )

    if data.deforestation:
        context.extend(
            ["deforestation", "fragmented habitat", "isolated habitat"]
        )

    if data.species_richness:
        context.append(
            f"{data.species_richness} species richness"
        )

        if data.species_richness.lower() == "low":
            context.append("low species richness")

    return " ".join(context)


# ---------------------------------------------------------
# KNOWLEDGE RETRIEVAL
# ---------------------------------------------------------

def retrieve_knowledge(text: str, top_k: int = 3):
    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        knowledge_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(similarities[index])

        if score > 0:
            item = KNOWLEDGE_BASE[index].copy()
            item["retrieval_score"] = round(score, 3)
            results.append(item)

    return results


# ---------------------------------------------------------
# MULTI-METRIC ENVIRONMENTAL REASONING
# ---------------------------------------------------------

def build_reasoning(data: EnvironmentalInput):
    observations = []

    low_carbon = (
        data.soil_organic_carbon is not None
        and data.soil_organic_carbon < 1
    )

    low_rainfall = (
        data.rainfall is not None
        and data.rainfall.lower() == "low"
    )

    irregular_rainfall = (
        data.rainfall is not None
        and data.rainfall.lower() == "irregular"
    )

    monoculture = (
        data.land_use is not None
        and "monoculture" in data.land_use.lower()
    )

    semi_arid = (
        data.region is not None
        and "semi-arid" in data.region.lower()
    )

    if low_carbon:
        observations.append(
            "Low soil organic carbon indicates limited organic matter "
            "and potentially reduced soil resilience."
        )

    if low_rainfall or irregular_rainfall:
        observations.append(
            "Limited or irregular rainfall increases water stress "
            "and makes moisture conservation important."
        )

    if monoculture:
        observations.append(
            "Monoculture provides lower structural and habitat diversity "
            "than a diversified production system."
        )

    if data.deforestation:
        observations.append(
            "Deforestation can reduce habitat availability and increase "
            "landscape fragmentation."
        )

    if data.pollution:
        observations.append(
            "Pollution adds environmental stress that may affect habitat "
            "quality and species survival."
        )

    if (
        data.species_richness
        and data.species_richness.lower() == "low"
    ):
        observations.append(
            "Low species richness suggests reduced biological diversity "
            "within the assessed system."
        )

    interactions = []

    if low_carbon and (low_rainfall or irregular_rainfall):
        interactions.append(
            "Low soil carbon and water stress interact: improving soil "
            "cover and organic matter can support a more resilient "
            "soil-water system."
        )

    if monoculture and low_carbon:
        interactions.append(
            "Monoculture combined with low soil carbon suggests that "
            "crop and vegetation diversification should be considered "
            "alongside soil-restoration measures."
        )

    if monoculture and (low_rainfall or semi_arid):
        interactions.append(
            "A low-rainfall monoculture system faces both climatic and "
            "ecological vulnerability, so diversification should account "
            "for drought tolerance as well as habitat value."
        )

    if data.deforestation and (
        data.species_richness
        and data.species_richness.lower() == "low"
    ):
        interactions.append(
            "Habitat loss together with low species richness indicates "
            "that restoring native vegetation and connectivity should "
            "be considered together."
        )

    if not observations:
        observations.append(
            "The supplied variables were assessed together to identify "
            "relevant environmental pressures."
        )

    return {
        "observed_pressures": observations,
        "cross_metric_interactions": interactions,
    }


# ---------------------------------------------------------
# API
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "system": "Darukaa Biodiversity Intelligence AI",
        "status": "running",
        "version": "1.1.0",
        "description": (
            "Knowledge-grounded environmental decision-support system"
        ),
        "knowledge_records": len(KNOWLEDGE_BASE),
        "endpoint": "/recommend",
        "interactive_docs": "/docs",
    }


@app.post("/recommend")
def recommend(data: EnvironmentalInput):
    missing = missing_information(data)

    # Ask for clarification if most core variables are absent.
    if len(missing) >= 2:
        return {
            "status": "needs_more_information",
            "message": (
                "I need additional environmental information before "
                "making a sufficiently grounded recommendation."
            ),
            "clarifying_questions": missing,
        }

    context = build_environmental_context(data)
    retrieved = retrieve_knowledge(context)

    if not retrieved:
        return {
            "status": "insufficient_knowledge",
            "message": (
                "The current knowledge base does not contain enough "
                "relevant evidence for this scenario."
            ),
            "recommendation": (
                "Collect additional site-specific environmental data "
                "before making a management decision."
            ),
        }

    multi_metric_reasoning = build_reasoning(data)

    recommendations = []

    for rank, item in enumerate(retrieved, start=1):
        recommendations.append(
            {
                "priority": rank,
                "recommendation": item["recommendation"],
                "scientific_reasoning": item["reasoning"],
                "impacted_metrics": item["metrics"],
                "time_horizon": item["time_horizon"],
                "evidence": item.get("evidence"),
                "evidence_source": item["source"],
                "source_url": item.get("source_url"),
                "retrieval_confidence": item["retrieval_score"],
            }
        )

    return {
        "status": "success",
        "environmental_context": context,
        "multi_metric_reasoning": multi_metric_reasoning,
        "recommendations": recommendations,
        "method": {
            "retrieval": (
                "TF-IDF vectorization with cosine-similarity ranking"
            ),
            "reasoning": (
                "Cross-variable environmental reasoning followed by "
                "evidence-backed recommendation retrieval"
            ),
            "knowledge_layer": (
                "Structured environmental knowledge records grounded "
                "in authoritative scientific sources"
            ),
        },
        "important_note": (
            "Recommendations are decision-support guidance and should "
            "be adapted using local ecological and site-specific data."
        ),
    }