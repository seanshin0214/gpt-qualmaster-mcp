#!/usr/bin/env python
"""RAG 검색 테스트 스크립트"""
import sys
sys.path.insert(0, '.')

from server import init_chromadb, search_chromadb

print('=== ChromaDB 초기화 테스트 ===')
result = init_chromadb()
print(f'ChromaDB 연결: {result}')

if result:
    print()
    print('=== RAG 검색 테스트 ===')
    results = search_chromadb('근거이론 코딩 방법', n_results=3)
    print(f'검색 결과 수: {len(results)}')

    for i, r in enumerate(results, 1):
        coll = r.get('collection', '?')
        title = r.get('title', '제목없음')[:50]
        content = r.get('content', '')[:100]
        print(f'{i}. [{coll}] {title}')
        print(f'   내용: {content}...')
        print()
else:
    print('ChromaDB 연결 실패!')
