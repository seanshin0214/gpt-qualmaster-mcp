# Changelog

## [1.1.0] - 2025-12-07

### Fixed
- **RAG 검색 기능 수정**: HttpClient에서 PersistentClient로 변경
  - 기존: `chromadb.HttpClient(port=8000)` - 외부 ChromaDB 서버 필요
  - 변경: `chromadb.PersistentClient(path="data/chroma_db")` - 자체 포함형

### Added
- `init_vectordb.py` - VectorDB 초기화 스크립트
  - 내장 지식(PARADIGMS, TRADITIONS 등)에서 20개 문서 생성
  - SentenceTransformer 'all-MiniLM-L6-v2' 임베딩 사용
- `QualMasterVectorStore` 클래스 - 지연 로딩 벡터 스토어

### Changed
- `search_knowledge` 도구가 실제 RAG 검색 수행
- 서버 상태 메시지에 벡터 스토어 상태 표시

## [1.0.0] - Initial Release
- 질적연구 방법론 지원 MCP 서버
- 20개 도구 제공
