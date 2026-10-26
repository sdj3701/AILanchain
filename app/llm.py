from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import get_settings


def get_llm(reasoning: bool = False) -> ChatOllama:
    """답변용 LLM. 기본은 thinking을 꺼서 빠르게 응답한다."""
    s = get_settings()
    return ChatOllama(
        model=s.llm_model,
        base_url=s.ollama_base_url,
        reasoning=reasoning,
        num_ctx=s.llm_num_ctx,
        keep_alive=s.llm_keep_alive,
        temperature=0.2,  # 사내 문서 Q&A는 창의성보다 일관성이 중요
    )


def get_embeddings() -> OllamaEmbeddings:
    s = get_settings()
    return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url)
