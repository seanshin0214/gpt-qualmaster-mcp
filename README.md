# GPT QualMaster MCP Server

> AI-Powered Qualitative Research & Conceptual Paper Writing Assistant for ChatGPT Desktop

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-2024--11--05-purple)](https://modelcontextprotocol.io)

## Overview

Dr. QualMaster는 질적연구자와 개념논문 작성자를 위한 GPT Desktop MCP 서버입니다.

### Features

- **12개 전문 도구** - 질적연구 지원 특화
- **4대 패러다임** - 실증주의, 후기실증주의, 비판이론, 구성주의
- **5대 질적연구 전통** - 현상학, 근거이론, 문화기술지, 내러티브, 사례연구
- **코딩 가이드** - 개방/축/선택 코딩, 주제분석
- **저널 가이드** - AMR, ASQ 스타일
- **R&R 지원** - 리젝션 패턴 진단 및 수정 전략

## Installation

```bash
# Clone
git clone https://github.com/seanshin0214/gpt-qualmaster-mcp.git
cd gpt-qualmaster-mcp

# Install dependencies
pip install -r requirements.txt

# Run
python server.py
```

## Usage

### 1. 서버 시작
```bash
python server.py
# Server running on http://127.0.0.1:8780
```

### 2. ngrok 터널 (개별 사용 시)
```bash
ngrok http 8780
```

### 3. ChatGPT Desktop 설정

| 항목 | 값 |
|------|-----|
| **Name** | `QualMaster` |
| **Description** | `정성연구 방법론 전문가. 질적 연구 패러다임, 연구 전통(현상학, 근거이론, 문화기술지 등), 코딩 가이드, 논문 심사 및 수정, 저널 투고 전략을 지원합니다.` |
| **URL** | `https://[your-ngrok-url]/mcp` (개별) 또는 `https://[gateway-ngrok-url]/qualmaster/mcp` (게이트웨이) |

#### Gateway 사용 시
[gpt-mcp-launcher](https://github.com/seanshin0214/gpt-mcp-launcher) 게이트웨이를 통해 여러 MCP 서버를 하나의 ngrok URL로 관리할 수 있습니다.

## 12 MCP Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | 질적연구 지식 검색 |
| `get_paradigm` | 연구 패러다임 상세 |
| `get_tradition` | 질적연구 전통 상세 |
| `suggest_methodology` | 방법론 추천 |
| `get_coding_guide` | 코딩 방법 가이드 |
| `assess_quality` | 품질 평가 기준 |
| `get_journal_guide` | 저널 투고 가이드 |
| `diagnose_rejection` | 리젝션 패턴 진단 |
| `conceptualize_idea` | 아이디어 개념화 |
| `develop_proposition` | 명제 개발 |
| `review_paper` | 논문 리뷰 |
| `guide_revision` | R&R 수정 가이드 |

## Supported Methodologies

### Paradigms
- Positivism (실증주의)
- Post-positivism (후기실증주의)
- Critical Theory (비판이론)
- Constructivism (구성주의)

### Qualitative Traditions
- Phenomenology (현상학)
- Grounded Theory (근거이론)
- Ethnography (문화기술지)
- Narrative Inquiry (내러티브)
- Case Study (사례연구)

### Coding Methods
- Open/Axial/Selective Coding
- In Vivo Coding
- Thematic Analysis (Braun & Clarke)

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Sean Shin** - Qualitative Research & AI Integration

---

*Built for qualitative researchers worldwide*
