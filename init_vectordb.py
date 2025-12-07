"""
QualMaster ChromaDB 초기화 스크립트
===================================
내장된 Knowledge Base를 ChromaDB에 벡터화
"""

import json
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# 경로 설정
BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"


# ============================================================================
# Knowledge Base (Embedded)
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
    "thematic_analysis": {
        "name": "주제분석 (Thematic Analysis)",
        "description": "Braun & Clarke의 6단계 주제분석",
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
    "lincoln_guba": {
        "name": "Lincoln & Guba (1985) 신뢰성 기준",
        "criteria": {
            "credibility": {
                "name": "신빙성 (Credibility)",
                "equivalent": "내적 타당도",
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
                "equivalent": "외적 타당도",
                "strategies": [
                    "풍부한 기술 (Thick description)",
                    "맥락 상세 기술",
                    "참여자 특성 명시"
                ]
            },
            "dependability": {
                "name": "의존가능성 (Dependability)",
                "equivalent": "신뢰도",
                "strategies": [
                    "감사 추적 (Audit trail)",
                    "탐구 과정 문서화",
                    "연구 일지 (Reflexive journal)"
                ]
            },
            "confirmability": {
                "name": "확인가능성 (Confirmability)",
                "equivalent": "객관성",
                "strategies": [
                    "감사 추적",
                    "삼각검증",
                    "반성적 성찰 (Reflexivity)"
                ]
            }
        }
    },
    "tracy": {
        "name": "Tracy (2010) 8가지 기준",
        "criteria": [
            "가치있는 주제 (Worthy Topic)",
            "풍부한 엄격성 (Rich Rigor)",
            "성실성 (Sincerity)",
            "신빙성 (Credibility)",
            "공명 (Resonance)",
            "의미있는 기여 (Significant Contribution)",
            "윤리성 (Ethics)",
            "의미있는 일관성 (Meaningful Coherence)"
        ]
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


def generate_documents() -> List[Dict]:
    """내장된 Knowledge Base에서 문서 생성"""
    documents = []
    doc_id = 0

    # Paradigms
    for key, p in PARADIGMS.items():
        content = f"""# {p['name']}

## 존재론 (Ontology)
{p['ontology']}

## 인식론 (Epistemology)
{p['epistemology']}

## 방법론 (Methodology)
{p['methodology']}

## 품질 기준
{', '.join(p['quality_criteria'])}

## 주요 학자
{', '.join(p['key_scholars'])}

## 한계
{p['limitations']}
"""
        documents.append({
            "id": f"paradigm_{key}",
            "content": content,
            "title": p['name'],
            "source": "paradigms",
            "category": "paradigm"
        })

    # Traditions
    for key, t in TRADITIONS.items():
        variants_text = "\n".join([f"- **{k}**: {v}" for k, v in t.get('variants', {}).items()])
        content = f"""# {t['name']}

## 연구 초점
{t['focus']}

## 데이터 수집
{t['data_collection']}

## 분석 방법
{t['analysis']}

## 표본 크기
{t['sample_size']}

## 주요 학자
{', '.join(t['key_scholars'])}

## 변형 (Variants)
{variants_text}
"""
        documents.append({
            "id": f"tradition_{key}",
            "content": content,
            "title": t['name'],
            "source": "traditions",
            "category": "tradition"
        })

    # Coding Types
    for key, c in CODING_TYPES.items():
        process_text = "\n".join([f"- {p}" for p in c.get('process', [])]) if 'process' in c else ""
        content = f"""# {c['name']}

## 설명
{c['description']}

## 절차
{process_text}

## 결과물
{c.get('output', '')}
"""
        documents.append({
            "id": f"coding_{key}",
            "content": content,
            "title": c['name'],
            "source": "coding",
            "category": "method"
        })

    # Quality Criteria
    for key, q in QUALITY_CRITERIA.items():
        if key == "lincoln_guba":
            criteria_text = ""
            for ck, cv in q['criteria'].items():
                strategies = "\n  ".join([f"- {s}" for s in cv['strategies']])
                criteria_text += f"\n### {cv['name']}\n양적연구 대응: {cv['equivalent']}\n전략:\n  {strategies}\n"
            content = f"""# {q['name']}

{criteria_text}
"""
        else:
            criteria_text = "\n".join([f"- {c}" for c in q['criteria']])
            content = f"""# {q['name']}

## 8가지 기준
{criteria_text}
"""
        documents.append({
            "id": f"quality_{key}",
            "content": content,
            "title": q['name'],
            "source": "quality",
            "category": "quality"
        })

    # Journals
    for key, j in JOURNALS.items():
        rejections = "\n".join([f"- {r}" for r in j.get('common_rejections', [])])
        tips = "\n".join([f"- {t}" for t in j.get('tips', [])])
        content = f"""# {j['name']}

## 초점
{j['focus']}

## 스타일
{j['style']}

## 주요 섹션
{', '.join(j.get('key_sections', []))}

## 흔한 리젝션 사유
{rejections}

## 투고 팁
{tips}
"""
        documents.append({
            "id": f"journal_{key}",
            "content": content,
            "title": j['name'],
            "source": "journals",
            "category": "journal"
        })

    # Rejection Patterns
    for key, r in REJECTION_PATTERNS.items():
        symptoms = "\n".join([f"- {s}" for s in r['symptoms']])
        solutions = "\n".join([f"- {s}" for s in r['solutions']])
        content = f"""# {r['name']}

## 증상
{symptoms}

## 해결 전략
{solutions}
"""
        documents.append({
            "id": f"rejection_{key}",
            "content": content,
            "title": r['name'],
            "source": "rejection_patterns",
            "category": "rejection"
        })

    return documents


def init_chromadb():
    """ChromaDB 초기화 및 데이터 저장"""
    print("\n" + "=" * 60)
    print("  QualMaster ChromaDB 초기화")
    print("=" * 60)

    # 디렉토리 생성
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    # SentenceTransformer 로드
    print("\n[1/4] SentenceTransformer 모델 로드...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # ChromaDB 클라이언트 생성
    print("\n[2/4] ChromaDB 클라이언트 생성...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection("qualmaster_knowledge")
    except:
        pass

    collection = client.create_collection(
        name="qualmaster_knowledge",
        metadata={"description": "QualMaster Knowledge Base"}
    )

    # 문서 생성
    print("\n[3/4] Knowledge Base에서 문서 생성...")
    documents = generate_documents()
    print(f"  -> {len(documents)} documents generated")

    # 임베딩 생성 및 저장
    print("\n[4/4] 임베딩 생성 및 저장...")

    ids = []
    contents = []
    metadatas = []

    for doc in documents:
        ids.append(doc["id"])
        contents.append(doc["content"])
        metadatas.append({
            "title": doc["title"],
            "source": doc["source"],
            "category": doc["category"]
        })

    # 배치로 임베딩 생성
    print("  Generating embeddings...")
    embeddings = encoder.encode(contents).tolist()

    # ChromaDB에 저장
    print("  Storing in ChromaDB...")
    collection.add(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\n" + "=" * 60)
    print(f"  완료! {len(documents)}개 문서가 ChromaDB에 저장됨")
    print(f"  경로: {CHROMA_PATH}")
    print("=" * 60 + "\n")

    # 테스트 검색
    print("테스트 검색: '현상학 연구 방법'")
    test_query = encoder.encode(["현상학 연구 방법"]).tolist()
    results = collection.query(query_embeddings=test_query, n_results=3)

    print("\n검색 결과:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n[{i+1}] {meta['title']} ({meta['source']})")
        print(f"    {doc[:150]}...")

    return True


if __name__ == "__main__":
    init_chromadb()
