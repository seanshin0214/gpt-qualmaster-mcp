# Changelog

## [1.1.1] - 2025-12-07 (Hotfix)

### Added
- **개념논문 작성 지식 추가** (6개 문서)
  - Gerring (1999) - What Makes a Concept Good? (8가지 기준)
  - Suddaby (2010) - Construct Clarity in Theories (AMR Editor's Comments)
  - Podsakoff et al. (2016) - Recommendations for Creating Better Concept Definitions
  - Whetten (1989) - What Constitutes a Theoretical Contribution?
  - Corley & Gioia (2011) - Building Theory about Theory Building
  - Conceptual Mechanisms - 심리적/사회적/구조적 메커니즘

### Changed
- VectorDB 문서 수: 20 → 26개
- RAG 검색 품질 검증 완료 (Podsakoff, Suddaby 등 정석 레퍼런스 반환 확인)

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
