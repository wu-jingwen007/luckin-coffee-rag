import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_app.embedder.embeddings import get_embedding_model
import numpy as np

print('Loading local model (first time is slow)...')
em = get_embedding_model()
print('Model loaded! Type:', type(em).__name__)

# 测试几个查询
test_queries = ['生椰拿铁', '生椰拿铁 原材料 制作', '融资 融资历程']

for q in test_queries:
    q_vec = em.embed_query(q)
    print()
    print('Query: %s' % q)
    print('  Embedding dim: %d, first 3: %s' % (len(q_vec), [round(v,4) for v in q_vec[:3]]))
    print('  SUCCESS')
