"""
GPT QualMaster MCP Server
=========================
AI-Powered Qualitative Research & Conceptual Paper Writing Assistant

원본: qualmaster-mcp-server (Claude MCP - TypeScript)
GPT Desktop용 Python FastAPI 포팅
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ChromaDB (optional - graceful fallback)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ChromaDB Configuration - PersistentClient (Local Storage)
BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"


class QualMasterVectorStore:
    """RAG 벡터 스토어 - ChromaDB PersistentClient 기반"""

    def __init__(self, chroma_path: str):
        self._client = None
        self._collection = None
        self._encoder = None
        self._chroma_path = chroma_path

    @property
    def encoder(self):
        if self._encoder is None and SENTENCE_TRANSFORMER_AVAILABLE:
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._encoder

    @property
    def collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = self._client.get_collection("qualmaster_knowledge")
            logger.info(f"Vector store loaded: {self._collection.count()} documents")
        return self._collection

    def search(self, query: str, n_results: int = 5, category: str = None) -> List[Dict]:
        """벡터 검색"""
        try:
            if not self.encoder:
                return []
            query_embedding = self.encoder.encode([query]).tolist()

            where_filter = None
            if category:
                where_filter = {"category": category}

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter
            )

            formatted = []
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                formatted.append({
                    "content": doc,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "category": meta.get("category", ""),
                    "rank": i + 1
                })
            return formatted
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def get_stats(self) -> Dict:
        """통계 반환"""
        try:
            return {
                "total_documents": self.collection.count(),
                "status": "connected"
            }
        except:
            return {"status": "disconnected"}


# Global vector store
vector_store: Optional[QualMasterVectorStore] = None


def init_chromadb():
    """Initialize ChromaDB PersistentClient"""
    global vector_store
    if not CHROMADB_AVAILABLE:
        logger.warning("ChromaDB not installed - RAG search disabled")
        return False
    try:
        chroma_path = str(CHROMA_PATH)
        if not CHROMA_PATH.exists():
            logger.warning(f"ChromaDB path not found: {chroma_path}")
            logger.warning("Run 'python init_vectordb.py' to initialize the vector database")
            return False

        vector_store = QualMasterVectorStore(chroma_path)
        stats = vector_store.get_stats()
        logger.info(f"ChromaDB PersistentClient connected: {stats['total_documents']} documents")
        return True
    except Exception as e:
        logger.warning(f"ChromaDB connection failed: {e}")
        vector_store = None
        return False

def safe_decode(text):
    """Safely decode text to UTF-8"""
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return text.decode('cp949')
            except:
                return text.decode('utf-8', errors='ignore')
    return str(text) if text else ""

def search_chromadb(query: str, n_results: int = 5, category: str = None) -> List[dict]:
    """Search ChromaDB for relevant documents using PersistentClient"""
    if not vector_store:
        return []

    try:
        results = vector_store.search(query, n_results, category)
        return results
    except Exception as e:
        logger.debug(f"Vector search failed: {e}")
        return []


# ============================================================================
# Knowledge Base (Embedded - No RAG dependency)
# ============================================================================

PARADIGMS = {
    "positivism": {
        "name": "실증주의 (Positivism)",
        "ontology": "단일한 객관적 실재 존재",
        "epistemology": "연구자와 대상의 분리, 객관적 관찰 가능",
        "methodology": "실험, 검증, 가설 검증",
        "quality_criteria": ["내적타당도", "외적타당도", "신뢰도", "객관성"],
        "key_scholars": ["Auguste Comte", "Émile Durkheim"],
        "limitations": "인간 경험의 복잡성과 맥락 무시 가능성"
    },
    "postpositivism": {
        "name": "후기실증주의 (Post-positivism)",
        "ontology": "실재 존재하나 완전히 파악 불가 (Critical Realism)",
        "epistemology": "객관성은 목표이나 편향 인정, 반증주의",
        "methodology": "수정된 실험, 삼각검증, 준실험설계",
        "quality_criteria": ["내적타당도", "외적타당도", "신뢰도", "객관성 (수정됨)"],
        "key_scholars": ["Karl Popper", "Thomas Kuhn"],
        "limitations": "여전히 객관주의적 가정 유지"
    },
    "critical_theory": {
        "name": "비판이론 (Critical Theory)",
        "ontology": "역사적으로 구성된 실재, 권력관계에 의해 형성",
        "epistemology": "연구자-대상 상호작용, 가치 개입 인정",
        "methodology": "이데올로기 비판, 참여적 행동연구, 담론분석",
        "quality_criteria": ["역사성", "침식성", "촉매적 타당성", "해방적 타당성"],
        "key_scholars": ["Theodor Adorno", "Jürgen Habermas", "Paulo Freire"],
        "limitations": "연구자의 정치적 입장이 연구에 과도하게 개입 가능"
    },
    "constructivism": {
        "name": "구성주의 (Constructivism)",
        "ontology": "다중 실재, 사회적으로 구성됨",
        "epistemology": "연구자-참여자 공동 구성, 해석학적 이해",
        "methodology": "해석학, 근거이론, 현상학, 내러티브",
        "quality_criteria": ["신뢰성", "전이가능성", "의존성", "확증성"],
        "key_scholars": ["Egon Guba", "Yvonna Lincoln", "John Creswell"],
        "limitations": "상대주의적 입장으로 인한 일반화 어려움"
    }
}

TRADITIONS = {
    "phenomenology": {
        "name": "현상학 (Phenomenology)",
        "focus": "체험의 본질 (essence of lived experience)",
        "data_collection": "심층 인터뷰, 참여 관찰",
        "analysis": "현상학적 환원, 본질 직관, 의미 단위 분석",
        "sample_size": "3-10명",
        "key_scholars": ["Husserl", "Heidegger", "Merleau-Ponty", "van Manen"],
        "variants": {
            "husserlian": "기술적 현상학 - 본질 기술에 집중",
            "heideggerian": "해석학적 현상학 - 존재론적 해석",
            "ipa": "해석학적 현상학적 분석 - 참여자의 의미 만들기"
        }
    },
    "grounded_theory": {
        "name": "근거이론 (Grounded Theory)",
        "focus": "현상을 설명하는 이론 개발",
        "data_collection": "인터뷰, 관찰, 문서 (이론적 표집)",
        "analysis": "개방코딩→축코딩→선택코딩, 지속적 비교",
        "sample_size": "20-30명 (포화 시점까지)",
        "key_scholars": ["Glaser", "Strauss", "Corbin", "Charmaz"],
        "variants": {
            "glaserian": "원전 GT - 출현, 연구자 중립성 강조",
            "straussian": "체계적 GT - 분석 절차 명확화",
            "charmaz": "구성주의 GT - 연구자의 해석적 역할 인정"
        }
    },
    "ethnography": {
        "name": "문화기술지 (Ethnography)",
        "focus": "문화 공유 집단의 패턴",
        "data_collection": "장기간 현장연구, 참여관찰, 인터뷰",
        "analysis": "기술, 분석, 해석",
        "sample_size": "문화 공유 집단 전체",
        "key_scholars": ["Clifford Geertz", "Bronisław Malinowski"],
        "variants": {
            "classical": "전통적 민족지 - 외부자 관점",
            "autoethnography": "자기 문화기술지 - 연구자 경험 중심",
            "critical": "비판적 민족지 - 권력관계 분석"
        }
    },
    "narrative": {
        "name": "내러티브 탐구 (Narrative Inquiry)",
        "focus": "개인의 이야기, 생애사",
        "data_collection": "생애사 인터뷰, 저널, 문서",
        "analysis": "줄거리 분석, 주제 분석, 구조 분석",
        "sample_size": "1-5명",
        "key_scholars": ["D. Jean Clandinin", "F. Michael Connelly"],
        "variants": {
            "biographical": "전기적 접근",
            "life_history": "생애사 연구",
            "oral_history": "구술사"
        }
    },
    "case_study": {
        "name": "사례연구 (Case Study)",
        "focus": "사례의 심층적 이해 (경계 지어진 체계)",
        "data_collection": "인터뷰, 관찰, 문서, 아카이브",
        "analysis": "사례 내 분석, 교차 사례 분석",
        "sample_size": "1-5 사례",
        "key_scholars": ["Robert Yin", "Robert Stake", "Kathleen Eisenhardt"],
        "variants": {
            "intrinsic": "본질적 - 사례 자체가 목적",
            "instrumental": "도구적 - 사례를 통한 이론 발전",
            "collective": "집합적 - 다중 사례 비교"
        }
    }
}

CODING_TYPES = {
    "open_coding": {
        "name": "개방코딩 (Open Coding)",
        "description": "데이터를 개념으로 분해하고 명명하는 과정",
        "process": ["라인별/문장별/단락별 분석", "개념 명명", "범주화"],
        "output": "개념 목록, 초기 범주"
    },
    "axial_coding": {
        "name": "축코딩 (Axial Coding)",
        "description": "범주 간 관계 구조화",
        "paradigm_model": {
            "causal_conditions": "인과적 조건",
            "phenomenon": "현상",
            "context": "맥락적 조건",
            "intervening": "중재적 조건",
            "strategies": "작용/상호작용 전략",
            "consequences": "결과"
        },
        "output": "패러다임 모형, 범주 관계도"
    },
    "selective_coding": {
        "name": "선택코딩 (Selective Coding)",
        "description": "핵심범주 선정 및 이론 통합",
        "process": ["스토리라인 작성", "핵심범주 확인", "이론적 틀 완성"],
        "output": "이론적 모형, 명제"
    },
    "invivo_coding": {
        "name": "인비보 코딩 (In Vivo Coding)",
        "description": "참여자의 실제 언어를 코드로 사용",
        "purpose": "참여자 관점 보존",
        "example": '"그냥 버티는 거죠" → 버티기'
    },
    "thematic_analysis": {
        "name": "주제분석 (Thematic Analysis)",
        "process": [
            "1. 데이터 친숙해지기",
            "2. 초기 코드 생성",
            "3. 주제 탐색",
            "4. 주제 검토",
            "5. 주제 정의 및 명명",
            "6. 보고서 작성"
        ],
        "key_scholars": ["Braun & Clarke (2006)"]
    }
}

QUALITY_CRITERIA = {
    "trustworthiness": {
        "credibility": {
            "name": "신뢰성 (Credibility)",
            "quantitative_equivalent": "내적 타당도",
            "strategies": [
                "장기간 참여 (Prolonged engagement)",
                "삼각검증 (Triangulation)",
                "동료 검토 (Peer debriefing)",
                "참여자 확인 (Member checking)",
                "부정사례 분석 (Negative case analysis)"
            ]
        },
        "transferability": {
            "name": "전이가능성 (Transferability)",
            "quantitative_equivalent": "외적 타당도",
            "strategies": [
                "풍부한 기술 (Thick description)",
                "맥락 상세 기술",
                "참여자 특성 명시"
            ]
        },
        "dependability": {
            "name": "의존성 (Dependability)",
            "quantitative_equivalent": "신뢰도",
            "strategies": [
                "감사 추적 (Audit trail)",
                "탐구 과정 문서화",
                "연구 일지 (Reflexive journal)"
            ]
        },
        "confirmability": {
            "name": "확증성 (Confirmability)",
            "quantitative_equivalent": "객관성",
            "strategies": [
                "감사 추적",
                "삼각검증",
                "반성적 성찰 (Reflexivity)"
            ]
        }
    }
}

JOURNALS = {
    "amr": {
        "name": "Academy of Management Review",
        "focus": "이론 개발, 개념 논문",
        "style": "가설 없음, 명제 중심",
        "key_sections": ["Introduction", "Theoretical Background", "Theory Development", "Discussion"],
        "common_rejections": [
            "So What? - 기여가 명확하지 않음",
            "Old Wine - 기존 이론의 재포장",
            "Logic Gaps - 논리적 비약",
            "Construct Confusion - 개념 혼란"
        ],
        "tips": [
            "첫 문단에서 흥미로운 Puzzle 제시",
            "기존 이론의 한계를 명확히",
            "새로운 개념/메커니즘 제안",
            "경쟁 가설과의 비교",
            "경계조건 명시"
        ]
    },
    "asq": {
        "name": "Administrative Science Quarterly",
        "focus": "경험적 연구 + 강한 이론",
        "style": "정성/정량 모두 가능",
        "requirements": [
            "새로운 이론적 통찰",
            "철저한 방법론",
            "풍부한 데이터"
        ]
    }
}

REJECTION_PATTERNS = {
    "so_what": {
        "name": "So What? 문제",
        "symptoms": ["기여 불명확", "실무적 함의 부족", "이론적 중요성 설명 부족"],
        "solutions": [
            "서론에서 연구 질문의 중요성 강조",
            "기존 이론/실무의 구체적 한계 제시",
            "연구 결과의 이론적/실무적 함의 명확화"
        ]
    },
    "old_wine": {
        "name": "Old Wine in New Bottles 문제",
        "symptoms": ["기존 연구와 차별점 불명확", "새로운 통찰 부족"],
        "solutions": [
            "기존 연구와의 명확한 차별점 제시",
            "새로운 맥락/조건에서의 적용",
            "기존 이론의 경계조건 탐색"
        ]
    },
    "logic_gap": {
        "name": "논리적 비약 문제",
        "symptoms": ["추론 과정 불완전", "데이터-결론 연결 약함"],
        "solutions": [
            "단계별 논리 전개",
            "대안 설명 고려",
            "추론의 근거 명시"
        ]
    }
}


# ============================================================================
# MCP Tools Definition
# ============================================================================

SERVER_INFO = {
    "name": "gpt-qualmaster-mcp",
    "version": "1.0.0",
    "description": "AI-Powered Qualitative Research & Conceptual Paper Writing Assistant for GPT Desktop"
}

TOOLS = [
    {
        "name": "search_knowledge",
        "description": """질적연구 및 개념논문 작성 지식을 검색합니다.

검색 가능한 주제:
- 연구 패러다임 (실증주의, 구성주의 등)
- 질적연구 전통 (현상학, 근거이론, 문화기술지 등)
- 코딩 방법 (개방코딩, 축코딩, 주제분석)
- 품질 기준 (신뢰성, 전이가능성)
- 저널 투고 (AMR, ASQ)
- R&R 전략 (리젝션 패턴, 수정 가이드)
        """,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 주제 또는 질문"},
                "category": {
                    "type": "string",
                    "enum": ["paradigms", "traditions", "coding", "quality", "journals", "rejection"],
                    "description": "검색 카테고리 (선택사항)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_paradigm",
        "description": "특정 연구 패러다임의 상세 정보를 반환합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paradigm": {
                    "type": "string",
                    "enum": ["positivism", "postpositivism", "critical_theory", "constructivism"],
                    "description": "패러다임 이름"
                }
            },
            "required": ["paradigm"]
        }
    },
    {
        "name": "get_tradition",
        "description": "특정 질적연구 전통의 상세 정보를 반환합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tradition": {
                    "type": "string",
                    "enum": ["phenomenology", "grounded_theory", "ethnography", "narrative", "case_study"],
                    "description": "질적연구 전통"
                }
            },
            "required": ["tradition"]
        }
    },
    {
        "name": "suggest_methodology",
        "description": "연구질문에 적합한 방법론을 추천합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "research_question": {"type": "string", "description": "연구질문"},
                "focus": {
                    "type": "string",
                    "enum": ["experience", "theory_building", "culture", "story", "case"],
                    "description": "연구 초점"
                }
            },
            "required": ["research_question"]
        }
    },
    {
        "name": "get_coding_guide",
        "description": "코딩 방법에 대한 가이드를 제공합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "coding_type": {
                    "type": "string",
                    "enum": ["open_coding", "axial_coding", "selective_coding", "invivo_coding", "thematic_analysis"],
                    "description": "코딩 유형"
                }
            },
            "required": ["coding_type"]
        }
    },
    {
        "name": "assess_quality",
        "description": """질적연구의 품질을 Lincoln & Guba + Tracy 기준으로 실제 평가합니다.

연구 설명과 사용 전략을 입력하면 100점 만점으로 점수를 산출합니다.
- Lincoln & Guba: 신빙성, 전이가능성, 의존가능성, 확인가능성
- Tracy: 가치있는 주제, 풍부한 엄격성, 성실성, 신빙성, 공명, 의미있는 기여, 윤리성, 의미있는 일관성""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "research_description": {
                    "type": "string",
                    "description": "연구 설명 (방법론, 데이터 수집, 분석 절차 등)"
                },
                "strategies_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "사용한 품질 전략 목록 (예: 삼각검증, 참여자확인, 감사추적 등)"
                },
                "criteria": {
                    "type": "string",
                    "enum": ["lincoln_guba", "tracy", "all"],
                    "description": "평가 기준 (기본값: all)"
                }
            },
            "required": ["research_description"]
        }
    },
    {
        "name": "get_journal_guide",
        "description": "저널 투고 가이드를 제공합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {
                    "type": "string",
                    "enum": ["amr", "asq"],
                    "description": "저널 이름"
                }
            },
            "required": ["journal"]
        }
    },
    {
        "name": "diagnose_rejection",
        "description": "리젝션 패턴을 진단하고 대응 전략을 제시합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rejection_type": {
                    "type": "string",
                    "enum": ["so_what", "old_wine", "logic_gap"],
                    "description": "리젝션 유형"
                }
            },
            "required": ["rejection_type"]
        }
    },
    {
        "name": "conceptualize_idea",
        "description": "연구 아이디어를 개념화하는 프레임워크를 제공합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "연구 아이디어"},
                "field": {"type": "string", "description": "연구 분야"}
            },
            "required": ["idea"]
        }
    },
    {
        "name": "develop_proposition",
        "description": "이론적 명제 개발을 지원합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "concept_a": {"type": "string", "description": "개념 A"},
                "concept_b": {"type": "string", "description": "개념 B"},
                "relationship": {
                    "type": "string",
                    "enum": ["positive", "negative", "moderation", "mediation"],
                    "description": "관계 유형"
                }
            },
            "required": ["concept_a", "concept_b"]
        }
    },
    {
        "name": "review_paper",
        "description": "논문 초안을 리뷰하고 피드백을 제공합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_section": {
                    "type": "string",
                    "enum": ["introduction", "literature", "method", "findings", "discussion"],
                    "description": "검토할 섹션"
                },
                "content": {"type": "string", "description": "검토할 내용"}
            },
            "required": ["paper_section", "content"]
        }
    },
    {
        "name": "guide_revision",
        "description": "R&R 수정 전략을 가이드합니다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reviewer_comment": {"type": "string", "description": "리뷰어 코멘트"},
                "comment_type": {
                    "type": "string",
                    "enum": ["major", "minor", "clarification"],
                    "description": "코멘트 유형"
                }
            },
            "required": ["reviewer_comment"]
        }
    }
]


# ============================================================================
# Tool Handlers
# ============================================================================

def handle_search_knowledge(args: dict) -> str:
    """지식 검색 - 내장 지식 + ChromaDB RAG 통합"""
    query = args.get("query", "").lower()
    original_query = args.get("query", "")
    category = args.get("category")

    results = []

    # 1. 내장 지식베이스 검색
    # 패러다임 검색
    if not category or category == "paradigms":
        for key, p in PARADIGMS.items():
            if query in key or query in p["name"].lower() or query in str(p).lower():
                results.append(f"**{p['name']}**\n- 존재론: {p['ontology']}\n- 인식론: {p['epistemology']}")

    # 전통 검색
    if not category or category == "traditions":
        for key, t in TRADITIONS.items():
            if query in key or query in t["name"].lower() or query in str(t).lower():
                results.append(f"**{t['name']}**\n- 초점: {t['focus']}\n- 분석: {t['analysis']}")

    # 코딩 검색
    if not category or category == "coding":
        for key, c in CODING_TYPES.items():
            if query in key or query in c["name"].lower() or query in str(c).lower():
                results.append(f"**{c['name']}**\n- {c['description']}")

    # 저널 검색
    if not category or category == "journals":
        for key, j in JOURNALS.items():
            if query in key or query in j["name"].lower():
                results.append(f"**{j['name']}**\n- 초점: {j['focus']}")

    # 리젝션 검색
    if not category or category == "rejection":
        for key, r in REJECTION_PATTERNS.items():
            if query in key or query in r["name"].lower():
                results.append(f"**{r['name']}**\n- 증상: {', '.join(r['symptoms'])}")

    # 2. ChromaDB RAG 검색 (추가 컨텍스트)
    rag_results = search_chromadb(original_query, n_results=5)
    rag_section = ""
    if rag_results:
        rag_section = "\n\n---\n\n## 📚 RAG 지식베이스 검색 결과\n\n"
        for i, r in enumerate(rag_results[:3], 1):
            content_preview = r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"]
            title = r.get("title", "")
            rag_section += f"### {i}. {title}\n{content_preview}\n\n"

    if results or rag_results:
        output = f"## '{original_query}' 검색 결과\n\n"
        if results:
            output += "\n\n---\n\n".join(results)
        output += rag_section
        return output
    else:
        return f"'{original_query}'에 대한 결과를 찾을 수 없습니다.\n\n사용 가능한 카테고리: paradigms, traditions, coding, quality, journals, rejection\n\n💡 ChromaDB 연결 상태: {'✅ 연결됨' if vector_store else '❌ 연결 안됨'}"


def handle_get_paradigm(args: dict) -> str:
    """패러다임 상세 정보"""
    paradigm = args.get("paradigm")
    if paradigm not in PARADIGMS:
        return f"알 수 없는 패러다임: {paradigm}\n사용 가능: {', '.join(PARADIGMS.keys())}"

    p = PARADIGMS[paradigm]
    return f"""## {p['name']}

### 존재론 (Ontology)
{p['ontology']}

### 인식론 (Epistemology)
{p['epistemology']}

### 방법론 (Methodology)
{p['methodology']}

### 품질 기준
{', '.join(p['quality_criteria'])}

### 주요 학자
{', '.join(p['key_scholars'])}

### 한계
{p['limitations']}
"""


def handle_get_tradition(args: dict) -> str:
    """질적연구 전통 상세 정보"""
    tradition = args.get("tradition")
    if tradition not in TRADITIONS:
        return f"알 수 없는 전통: {tradition}\n사용 가능: {', '.join(TRADITIONS.keys())}"

    t = TRADITIONS[tradition]
    variants_text = "\n".join([f"- **{k}**: {v}" for k, v in t.get("variants", {}).items()])

    return f"""## {t['name']}

### 연구 초점
{t['focus']}

### 데이터 수집
{t['data_collection']}

### 분석 방법
{t['analysis']}

### 표본 크기
{t['sample_size']}

### 주요 학자
{', '.join(t['key_scholars'])}

### 변형 (Variants)
{variants_text}
"""


def handle_suggest_methodology(args: dict) -> str:
    """방법론 추천"""
    rq = args.get("research_question", "")
    focus = args.get("focus")

    suggestions = []

    # 키워드 기반 추천
    rq_lower = rq.lower()

    if "경험" in rq or "체험" in rq or "experience" in rq_lower:
        suggestions.append(("현상학", "개인의 체험과 본질 탐구에 적합", "phenomenology"))

    if "과정" in rq or "어떻게" in rq or "process" in rq_lower or "how" in rq_lower:
        suggestions.append(("근거이론", "과정 설명 및 이론 개발에 적합", "grounded_theory"))

    if "문화" in rq or "집단" in rq or "culture" in rq_lower:
        suggestions.append(("문화기술지", "문화 공유 집단 연구에 적합", "ethnography"))

    if "이야기" in rq or "생애" in rq or "story" in rq_lower or "life" in rq_lower:
        suggestions.append(("내러티브 탐구", "개인 이야기와 생애사 연구에 적합", "narrative"))

    if "사례" in rq or "case" in rq_lower or "왜" in rq or "why" in rq_lower:
        suggestions.append(("사례연구", "맥락 내 심층 분석에 적합", "case_study"))

    if not suggestions:
        suggestions = [
            ("현상학", "체험의 본질 탐구", "phenomenology"),
            ("근거이론", "이론 개발", "grounded_theory"),
            ("사례연구", "심층 분석", "case_study")
        ]

    output = f"## 연구질문 분석\n\n**질문**: {rq}\n\n### 추천 방법론\n\n"
    for name, reason, key in suggestions:
        t = TRADITIONS[key]
        output += f"#### {name}\n- **적합 이유**: {reason}\n- **표본 크기**: {t['sample_size']}\n- **분석 방법**: {t['analysis']}\n\n"

    return output


def handle_get_coding_guide(args: dict) -> str:
    """코딩 가이드"""
    coding_type = args.get("coding_type")
    if coding_type not in CODING_TYPES:
        return f"알 수 없는 코딩 유형: {coding_type}\n사용 가능: {', '.join(CODING_TYPES.keys())}"

    c = CODING_TYPES[coding_type]
    output = f"## {c['name']}\n\n{c['description']}\n\n"

    if "process" in c:
        output += "### 절차\n" + "\n".join([f"- {p}" for p in c["process"]]) + "\n\n"

    if "output" in c:
        output += f"### 결과물\n{c['output']}\n\n"

    if "paradigm_model" in c:
        output += "### 패러다임 모형\n"
        for k, v in c["paradigm_model"].items():
            output += f"- **{k}**: {v}\n"

    return output


def normalize_text(text: str) -> str:
    """텍스트 정규화 - 띄어쓰기, 언더스코어 등을 무시하고 비교"""
    import re
    normalized = text.lower()
    normalized = re.sub(r'[\s_\-]', '', normalized)  # 공백, 언더스코어, 하이픈 제거
    normalized = normalized.replace('검증', '검토')  # 검증과 검토를 동일하게 처리
    return normalized


def assess_lincoln_guba(description: str, strategies: List[str]) -> List[dict]:
    """Lincoln & Guba 기준 평가"""
    lower_desc = description.lower()
    normalized_desc = normalize_text(description)
    normalized_strategies = [normalize_text(s) for s in strategies]

    criteria = [
        {
            "criterion": "credibility",
            "korean": "신빙성 (Credibility)",
            "strategies": [
                {
                    "name": "prolonged_engagement",
                    "korean": "장기적 관여",
                    "keywords": ["장기", "오랜기간", "prolonged", "7일", "14일", "집중적관여", "지속적"]
                },
                {
                    "name": "triangulation",
                    "korean": "삼각화/삼각검증",
                    "keywords": ["삼각화", "삼각검증", "triangulation", "다중자료", "3중", "인터뷰+저널", "다중출처"]
                },
                {
                    "name": "peer_debriefing",
                    "korean": "동료 검토",
                    "keywords": ["동료검토", "동료검증", "peer", "debriefing", "동료연구자"]
                },
                {
                    "name": "member_checking",
                    "korean": "참여자 확인",
                    "keywords": ["참여자확인", "membercheck", "memberchecking", "참여자검토", "2단계확인"]
                },
                {
                    "name": "negative_case",
                    "korean": "부정적 사례 분석",
                    "keywords": ["부정적사례", "negativecase", "반증", "방해경험", "부정사례"]
                }
            ]
        },
        {
            "criterion": "transferability",
            "korean": "전이가능성 (Transferability)",
            "strategies": [
                {
                    "name": "thick_description",
                    "korean": "두꺼운 기술",
                    "keywords": ["두꺼운기술", "thickdescription", "상세기술", "풍부한기술"]
                },
                {
                    "name": "purposeful_sampling",
                    "korean": "목적적 표본추출",
                    "keywords": ["목적적", "purposeful", "의도적표집", "목적표집", "목적적표본", "목적표본"]
                },
                {
                    "name": "context_description",
                    "korean": "맥락 기술",
                    "keywords": ["맥락", "context", "배경", "상황기술", "맥락상세", "맥락체크리스트"]
                }
            ]
        },
        {
            "criterion": "dependability",
            "korean": "의존가능성 (Dependability)",
            "strategies": [
                {
                    "name": "audit_trail",
                    "korean": "감사 추적",
                    "keywords": ["감사추적", "audittrail", "연구일지", "감사로그", "추적로그"]
                },
                {
                    "name": "code_recode",
                    "korean": "코드-재코드",
                    "keywords": ["재코드", "recode", "반복코딩", "코드재코드", "일치율", "코딩일치"]
                },
                {
                    "name": "peer_examination",
                    "korean": "동료 검증",
                    "keywords": ["동료검증", "동료검토", "peerexamination", "동료심사"]
                }
            ]
        },
        {
            "criterion": "confirmability",
            "korean": "확인가능성 (Confirmability)",
            "strategies": [
                {
                    "name": "reflexivity",
                    "korean": "반성성/성찰",
                    "keywords": ["반성", "reflexiv", "성찰", "반성적저널", "위치성", "저널링"]
                },
                {
                    "name": "audit_trail",
                    "korean": "감사 추적",
                    "keywords": ["감사추적", "audittrail", "감사로그"]
                },
                {
                    "name": "triangulation",
                    "korean": "삼각화/삼각검증",
                    "keywords": ["삼각화", "삼각검증", "triangulation", "3중"]
                }
            ]
        }
    ]

    results = []
    for c in criteria:
        applied = []
        missing = []

        for strategy in c["strategies"]:
            # description에서 키워드 찾기
            found_in_desc = any(
                normalize_text(k) in normalized_desc or k.lower() in lower_desc
                for k in strategy["keywords"]
            )

            # strategies_used 배열에서 찾기
            found_in_strategies = any(
                any(normalize_text(k) in s or s in normalize_text(k) for k in strategy["keywords"])
                for s in normalized_strategies
            )

            if found_in_desc or found_in_strategies:
                applied.append(strategy["korean"])
            else:
                missing.append(strategy["korean"])

        score = round((len(applied) / len(c["strategies"])) * 25)

        results.append({
            "criterion": c["criterion"],
            "korean": c["korean"],
            "score": score,
            "max_score": 25,
            "strategies_applied": applied,
            "missing_strategies": missing,
            "recommendations": [f"{m} 전략을 추가로 적용하세요" for m in missing] if missing else []
        })

    return results


def assess_tracy(description: str, strategies: List[str]) -> List[dict]:
    """Tracy 8가지 기준 평가"""
    lower_desc = description.lower()
    normalized_desc = normalize_text(description)
    normalized_strategies = [normalize_text(s) for s in strategies]

    criteria = [
        {
            "criterion": "worthy_topic",
            "korean": "가치있는 주제",
            "indicators": [
                "중요", "시의적절", "필요", "기여", "문제", "의미", "가치",
                "새로운현상", "AI", "리더", "의사결정", "탐구", "연구목적"
            ]
        },
        {
            "criterion": "rich_rigor",
            "korean": "풍부한 엄격성",
            "indicators": [
                "충분한", "다양한", "적절한", "체계적", "면밀한", "엄격",
                "IPA", "6단계", "다중사례", "심층", "분석절차", "브라케팅"
            ]
        },
        {
            "criterion": "sincerity",
            "korean": "성실성",
            "indicators": [
                "반성", "성찰", "한계", "투명", "정직", "위치성",
                "반성적저널", "저널링", "솔직"
            ]
        },
        {
            "criterion": "credibility",
            "korean": "신빙성",
            "indicators": [
                "삼각", "참여자확인", "두꺼운기술", "구체적", "검증",
                "membercheck", "삼각검증", "동료검토"
            ]
        },
        {
            "criterion": "resonance",
            "korean": "공명",
            "indicators": [
                "전이", "일반화", "독자", "영향", "감동", "공감",
                "경험", "의미", "본질", "통찰"
            ]
        },
        {
            "criterion": "significant_contribution",
            "korean": "의미있는 기여",
            "indicators": [
                "기여", "확장", "새로운", "발전", "함의", "이론적",
                "실무적", "통찰", "제안"
            ]
        },
        {
            "criterion": "ethics",
            "korean": "윤리성",
            "indicators": [
                "윤리", "동의", "익명", "보호", "IRB", "승인",
                "동의서", "철회", "민감정보", "익명화"
            ]
        },
        {
            "criterion": "meaningful_coherence",
            "korean": "의미있는 일관성",
            "indicators": [
                "일관", "연결", "목적", "방법론", "통합", "적합",
                "IPA", "현상학", "연구질문", "분석"
            ]
        }
    ]

    results = []
    for c in criteria:
        # description과 strategies 모두에서 indicator 찾기
        found_indicators = [
            ind for ind in c["indicators"]
            if normalize_text(ind) in normalized_desc or
               ind.lower() in lower_desc or
               any(normalize_text(ind) in s for s in normalized_strategies)
        ]

        missing_indicators = [
            ind for ind in c["indicators"]
            if normalize_text(ind) not in normalized_desc and
               ind.lower() not in lower_desc and
               not any(normalize_text(ind) in s for s in normalized_strategies)
        ]

        # 점수 계산 - 최소 1개만 매치되어도 부분 점수 부여
        match_ratio = len(found_indicators) / len(c["indicators"])
        score = round(match_ratio * 13)

        results.append({
            "criterion": c["criterion"],
            "korean": c["korean"],
            "score": score,
            "max_score": 13,
            "strategies_applied": found_indicators,
            "missing_strategies": missing_indicators[:3],  # 상위 3개만 표시
            "recommendations": [f"{c['korean']} 관련 내용을 보강하세요"] if score < 10 and len(found_indicators) < 3 else []
        })

    return results


def get_grade(percentage: float) -> str:
    """등급 계산"""
    if percentage >= 90:
        return "A (우수)"
    elif percentage >= 80:
        return "B (양호)"
    elif percentage >= 70:
        return "C (보통)"
    elif percentage >= 60:
        return "D (미흡)"
    else:
        return "F (개선 필요)"


def get_priority_actions(assessments: List[dict]) -> List[str]:
    """우선 조치 사항"""
    return [
        f"{a['korean']} 개선: {a['recommendations'][0] if a['recommendations'] else '전략 추가 필요'}"
        for a in assessments
        if a['score'] / a['max_score'] < 0.5
    ][:3]


def handle_assess_quality(args: dict) -> str:
    """품질 평가 - Lincoln & Guba + Tracy 기준으로 실제 점수 산출"""
    research_description = args.get("research_description", "")
    strategies_used = args.get("strategies_used", [])
    criteria = args.get("criteria", "all")

    if not research_description:
        return "연구 설명(research_description)을 입력해주세요."

    assessments = []

    if criteria == "lincoln_guba" or criteria == "all":
        assessments.extend(assess_lincoln_guba(research_description, strategies_used))

    if criteria == "tracy" or criteria == "all":
        assessments.extend(assess_tracy(research_description, strategies_used))

    # 전체 점수 계산
    total_score = sum(a["score"] for a in assessments)
    max_score = sum(a["max_score"] for a in assessments)
    overall_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

    # 강점/약점 식별
    strengths = [a["korean"] for a in assessments if a["score"] / a["max_score"] >= 0.7]
    weaknesses = [a["korean"] for a in assessments if a["score"] / a["max_score"] < 0.5]

    # 결과 구성
    result = {
        "criteria_used": criteria,
        "input_summary": {
            "description_length": len(research_description),
            "strategies_reported": len(strategies_used)
        },
        "overall_assessment": {
            "score": f"{total_score}/{max_score}",
            "percentage": f"{overall_percentage:.1f}%",
            "grade": get_grade(overall_percentage)
        },
        "detailed_assessment": [
            {
                "criterion": a["criterion"],
                "korean": a["korean"],
                "score": f"{a['score']}/{a['max_score']}",
                "strategies_applied": a["strategies_applied"],
                "missing_strategies": a["missing_strategies"],
                "recommendations": a["recommendations"]
            }
            for a in assessments
        ],
        "summary": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "priority_actions": get_priority_actions(assessments)
        },
        "quality_enhancement_guide": {
            "immediate_actions": [
                "연구 설계 단계에서 품질 전략을 계획하세요",
                "연구 일지를 꾸준히 작성하세요",
                "동료 연구자와 정기적으로 토론하세요"
            ],
            "during_data_collection": [
                "참여자와 충분한 라포를 형성하세요",
                "면담 후 즉시 메모를 작성하세요",
                "다양한 자료원을 활용하세요"
            ],
            "during_analysis": [
                "코딩의 일관성을 검토하세요",
                "참여자 확인(member checking)을 실시하세요",
                "부정적 사례를 적극적으로 찾으세요"
            ],
            "writing_phase": [
                "두꺼운 기술로 맥락을 풍부하게 제시하세요",
                "연구자의 위치성을 명시하세요",
                "한계를 솔직하게 논의하세요"
            ]
        }
    }

    # JSON 형식으로 반환 (가독성 있게)
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


def handle_get_journal_guide(args: dict) -> str:
    """저널 가이드"""
    journal = args.get("journal")
    if journal not in JOURNALS:
        return f"알 수 없는 저널: {journal}\n사용 가능: {', '.join(JOURNALS.keys())}"

    j = JOURNALS[journal]
    output = f"## {j['name']}\n\n"
    output += f"**초점**: {j['focus']}\n\n"
    output += f"**스타일**: {j['style']}\n\n"

    if "key_sections" in j:
        output += "### 주요 섹션\n" + ", ".join(j['key_sections']) + "\n\n"

    if "common_rejections" in j:
        output += "### 흔한 리젝션 사유\n"
        for r in j['common_rejections']:
            output += f"- {r}\n"
        output += "\n"

    if "tips" in j:
        output += "### 투고 팁\n"
        for t in j['tips']:
            output += f"- {t}\n"

    return output


def handle_diagnose_rejection(args: dict) -> str:
    """리젝션 진단"""
    rejection_type = args.get("rejection_type")
    if rejection_type not in REJECTION_PATTERNS:
        return f"알 수 없는 리젝션 유형: {rejection_type}\n사용 가능: {', '.join(REJECTION_PATTERNS.keys())}"

    r = REJECTION_PATTERNS[rejection_type]
    output = f"## {r['name']}\n\n"
    output += "### 증상\n" + "\n".join([f"- {s}" for s in r['symptoms']]) + "\n\n"
    output += "### 해결 전략\n" + "\n".join([f"- {s}" for s in r['solutions']])

    return output


def handle_conceptualize_idea(args: dict) -> str:
    """아이디어 개념화"""
    idea = args.get("idea", "")
    field = args.get("field", "경영학")

    return f"""## 연구 아이디어 개념화

### 입력 아이디어
{idea}

### 분야
{field}

### 개념화 프레임워크

#### 1. 핵심 개념 추출
- 주요 변수/개념은 무엇인가?
- 기존 문헌에서 어떻게 정의되는가?

#### 2. 이론적 틀
- 어떤 이론적 렌즈로 볼 것인가?
- 기존 이론의 한계는?

#### 3. 연구 질문 형성
- 경험적 질문 vs 개념적 질문?
- What/How/Why 중 어느 유형?

#### 4. 기대 기여
- 이론적 기여: 새로운 개념? 관계? 경계조건?
- 실무적 기여: 어떤 시사점?

### 다음 단계
1. 핵심 개념 정의 및 문헌 검토
2. 이론적 긴장 또는 Puzzle 식별
3. 연구 질문 정교화
"""


def handle_develop_proposition(args: dict) -> str:
    """명제 개발"""
    concept_a = args.get("concept_a", "A")
    concept_b = args.get("concept_b", "B")
    relationship = args.get("relationship", "positive")

    rel_templates = {
        "positive": f"{concept_a}이 높을수록 {concept_b}도 높아진다.",
        "negative": f"{concept_a}이 높을수록 {concept_b}는 낮아진다.",
        "moderation": f"{concept_a}와 종속변수의 관계는 {concept_b}에 의해 조절된다.",
        "mediation": f"{concept_a}은 {concept_b}를 통해 결과변수에 영향을 미친다."
    }

    return f"""## 이론적 명제 개발

### 개념
- **개념 A**: {concept_a}
- **개념 B**: {concept_b}
- **관계 유형**: {relationship}

### 명제 초안
**Proposition**: {rel_templates.get(relationship, f'{concept_a}과 {concept_b}는 관련이 있다.')}

### 명제 정교화 가이드

#### 1. 메커니즘 설명
- 왜 이 관계가 존재하는가?
- 어떤 과정을 통해 연결되는가?

#### 2. 경계조건 명시
- 언제 이 관계가 성립하는가?
- 어떤 상황에서 약화/강화되는가?

#### 3. 경쟁 설명 고려
- 대안적 설명은 무엇인가?
- 왜 그 설명보다 이 설명이 나은가?

#### 4. 검증 가능성
- 어떻게 경험적으로 검증할 수 있는가?
- 어떤 데이터가 필요한가?
"""


def handle_review_paper(args: dict) -> str:
    """논문 리뷰"""
    section = args.get("paper_section", "")
    content = args.get("content", "")

    review_guides = {
        "introduction": """
### Introduction 검토 기준

1. **Hook**: 첫 문장이 주의를 끄는가?
2. **Puzzle/Gap**: 연구 문제가 명확한가?
3. **Significance**: 왜 이 연구가 중요한가?
4. **Preview**: 연구 접근법이 소개되는가?
5. **Contribution**: 기여가 명확히 예고되는가?
""",
        "literature": """
### Literature Review 검토 기준

1. **Coverage**: 주요 문헌을 포함하는가?
2. **Synthesis**: 단순 나열이 아닌 통합인가?
3. **Gap Identification**: 문헌의 한계가 명확한가?
4. **Theoretical Foundation**: 이론적 기반이 견고한가?
""",
        "method": """
### Method 검토 기준

1. **Paradigm Fit**: 연구 질문과 방법론이 일치하는가?
2. **Sampling**: 표집 전략이 적절한가?
3. **Data Collection**: 데이터 수집이 철저한가?
4. **Analysis**: 분석 절차가 명확한가?
5. **Rigor**: 신뢰성 확보 전략이 있는가?
""",
        "findings": """
### Findings 검토 기준

1. **Evidence**: 주장에 충분한 증거가 있는가?
2. **Quotes**: 인용이 적절히 사용되는가?
3. **Organization**: 구조가 논리적인가?
4. **Saturation**: 주요 주제가 포화에 도달했는가?
""",
        "discussion": """
### Discussion 검토 기준

1. **Interpretation**: 결과 해석이 적절한가?
2. **Contribution**: 이론적 기여가 명확한가?
3. **Limitations**: 한계가 솔직히 인정되는가?
4. **Implications**: 함의가 구체적인가?
5. **Future Research**: 향후 연구 방향이 제시되는가?
"""
    }

    guide = review_guides.get(section, "선택한 섹션에 대한 가이드가 없습니다.")

    return f"""## {section.upper()} 섹션 리뷰

### 검토 대상 내용
```
{content[:500]}{'...' if len(content) > 500 else ''}
```

{guide}

### 일반 피드백 프레임워크

**강점 확인**: 잘 된 부분은?
**개선 필요**: 보완이 필요한 부분은?
**구체적 제안**: 어떻게 개선할 수 있는가?
"""


def handle_guide_revision(args: dict) -> str:
    """R&R 가이드"""
    comment = args.get("reviewer_comment", "")
    comment_type = args.get("comment_type", "major")

    return f"""## R&R 수정 가이드

### 리뷰어 코멘트
```
{comment}
```

### 코멘트 유형
**{comment_type.upper()}**

### 대응 전략

#### 1. 코멘트 분석
- 리뷰어가 원하는 것은 무엇인가?
- 구체적 수정? 설명 추가? 데이터 보강?

#### 2. 대응 옵션
- **수용**: 코멘트를 그대로 반영
- **부분 수용**: 일부만 반영하고 이유 설명
- **반박**: 정중하게 반론 (근거 필수)

#### 3. 응답 작성
- 감사 표현으로 시작
- 구체적 수정 내용 명시
- 페이지/라인 번호 포함

#### 4. 수정 팁 ({comment_type})
{"- 신중하고 철저한 수정 필요\n- 추가 분석이나 데이터 보강 고려\n- 이론적 논거 강화" if comment_type == "major" else "- 간단한 수정으로 해결 가능\n- 명확한 설명 추가" if comment_type == "minor" else "- 설명만 추가하면 됨\n- 본문 수정 없이 해명 가능"}

### 응답 템플릿
```
We thank the reviewer for this valuable comment. [감사]

[구체적 대응 내용]

We have revised the manuscript accordingly. Please see [section/page] for the updated version.
```
"""


# ============================================================================
# Main Tool Handler
# ============================================================================

async def handle_tool_call(name: str, arguments: dict) -> dict:
    """도구 호출 처리"""
    handlers = {
        "search_knowledge": handle_search_knowledge,
        "get_paradigm": handle_get_paradigm,
        "get_tradition": handle_get_tradition,
        "suggest_methodology": handle_suggest_methodology,
        "get_coding_guide": handle_get_coding_guide,
        "assess_quality": handle_assess_quality,
        "get_journal_guide": handle_get_journal_guide,
        "diagnose_rejection": handle_diagnose_rejection,
        "conceptualize_idea": handle_conceptualize_idea,
        "develop_proposition": handle_develop_proposition,
        "review_paper": handle_review_paper,
        "guide_revision": handle_guide_revision
    }

    handler = handlers.get(name)
    if handler:
        result = handler(arguments)
        return {"content": [{"type": "text", "text": result}]}
    else:
        return {"content": [{"type": "text", "text": f"알 수 없는 도구: {name}"}], "isError": True}


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("GPT QualMaster MCP Server Starting")
    logger.info("12 tools available for qualitative research")

    # Initialize ChromaDB
    if init_chromadb():
        logger.info("✅ ChromaDB RAG search enabled")
    else:
        logger.warning("⚠️ ChromaDB not available - using embedded knowledge only")

    logger.info("=" * 50)
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="GPT QualMaster MCP",
    description="AI-Powered Qualitative Research & Conceptual Paper Writing Assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "running",
        "server": SERVER_INFO,
        "tools_count": len(TOOLS)
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "tools": len(TOOLS)}


@app.get("/mcp")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for GPT MCP connections"""
    # Get the base URL from the request
    host = request.headers.get("host", "localhost:8780")
    scheme = request.headers.get("x-forwarded-proto", "http")
    base_url = f"{scheme}://{host}"

    async def event_generator():
        # First, send the endpoint event (MCP SSE protocol requirement)
        yield f"event: endpoint\ndata: {base_url}/mcp\n\n"

        # Send server info as a message
        init_event = {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {}}
            },
            "id": 0
        }
        yield f"event: message\ndata: {json.dumps(init_event)}\n\n"

        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield f": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()

        if body.get("method") == "initialize":
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": SERVER_INFO,
                    "capabilities": {"tools": {}}
                },
                "id": body.get("id")
            })

        elif body.get("method") == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": {"tools": TOOLS},
                "id": body.get("id")
            })

        elif body.get("method") == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            # Defensive coding: arguments가 None이거나 없는 경우 빈 딕셔너리로 처리
            arguments = params.get("arguments")
            if arguments is None or not isinstance(arguments, dict):
                arguments = {}
            result = await handle_tool_call(tool_name, arguments)
            return JSONResponse({
                "jsonrpc": "2.0",
                "result": result,
                "id": body.get("id")
            })

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": "Method not found"},
                "id": body.get("id")
            })

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse({
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": str(e)},
            "id": None
        }, status_code=400)


def main():
    print("\n" + "=" * 60)
    print("  GPT QualMaster MCP Server v1.0.0")
    print("  AI-Powered Qualitative Research Assistant")
    print("=" * 60)
    print("  URL: http://127.0.0.1:8780")
    print("  ngrok: ngrok http 8780")
    print("-" * 60)
    print("  12 Tools:")
    for t in TOOLS:
        print(f"    - {t['name']}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8780, log_level="info")


if __name__ == "__main__":
    main()
