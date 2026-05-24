"""
api.py — Endpoint FastAPI para el motor de búsqueda semántica (Bonus)

Uso:
    uvicorn api:app --reload

Endpoints:
    GET /buscar?q=<query>&n=<n_resultados>
    GET /health
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from buscar import buscar                     # reutiliza la función del lab

app = FastAPI(
    title="Motor de Búsqueda Semántica",
    description="API REST sobre embeddings OpenAI + ChromaDB",
    version="1.0.0",
)


# ── Modelos de respuesta ─────────────────────────────────────────────────────

class Resultado(BaseModel):
    id: str
    titulo: str
    contenido: str
    score: float


class RespuestaBusqueda(BaseModel):
    query: str
    n_resultados: int
    resultados: list[Resultado]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/buscar", response_model=RespuestaBusqueda)
def endpoint_buscar(
    q: str = Query(..., description="Texto de búsqueda", min_length=1),
    n: int = Query(3, description="Número de resultados", ge=1, le=10),
):
    """
    Busca artículos similares a la query usando embeddings semánticos.

    - **q**: texto de la búsqueda
    - **n**: número de resultados a devolver (1-10, por defecto 3)
    """
    try:
        resultados = buscar(q, n_resultados=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RespuestaBusqueda(
        query=q,
        n_resultados=len(resultados),
        resultados=[Resultado(**r) for r in resultados],
    )
