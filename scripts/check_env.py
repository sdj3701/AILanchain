"""0단계 환경 점검: LLM, 임베딩, Chroma가 모두 동작하는지 확인한다.

실행: .venv\\Scripts\\python.exe -m scripts.check_env
"""
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.llm import get_embeddings, get_llm


def check_llm():
    llm = get_llm()
    start = time.perf_counter()
    answer = llm.invoke("연차휴가가 무엇인지 한 문장으로 설명해줘.")
    print(f"[LLM] {time.perf_counter() - start:.2f}초 | {answer.content}")


def check_embeddings():
    vec = get_embeddings().embed_query("테스트 문장")
    print(f"[임베딩] 차원 수: {len(vec)}")


def check_chroma():
    docs = [
        Document(page_content="신입사원의 연차휴가는 입사 1년 미만일 때 매월 1일씩 발생한다.", metadata={"source": "인사규정.pdf"}),
        Document(page_content="식대는 월 20만원까지 비과세로 지급된다.", metadata={"source": "복리후생.pdf"}),
        Document(page_content="사내 와이파이 비밀번호는 IT 헬프데스크에 문의한다.", metadata={"source": "IT안내.pdf"}),
    ]
    store = Chroma.from_documents(docs, get_embeddings())  # 점검용이라 메모리에만 저장
    hit, score = store.similarity_search_with_score("밥값 지원은 얼마야?", k=1)[0]
    print(f"[Chroma] 검색 결과: {hit.metadata['source']} (거리 {score:.3f})")


if __name__ == "__main__":
    check_llm()
    check_embeddings()
    check_chroma()
    print("환경 점검 완료")
