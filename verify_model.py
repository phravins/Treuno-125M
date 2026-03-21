"""Antigravity component verification script."""
import sys
sys.path.insert(0, 'd:/MODEL')

print('=== Antigravity Component Import Verification ===')

# 1. AG-Retrieve
from antigravity.ag_retrieve import AGRetrieve, SOURCE_TIERS
ag_r = AGRetrieve()
assert ag_r.top_k_retrieve == 10 and ag_r.top_k_final == 3
required_tiers = ['github', 'stackoverflow', 'mdn', 'pypi_npm', 'lang_refs', 'arxiv', 'web']
assert all(t in SOURCE_TIERS for t in required_tiers)
# TTL checks from spec
assert SOURCE_TIERS['github']['cache_ttl'] == 3600         # 1 hour
assert SOURCE_TIERS['stackoverflow']['cache_ttl'] == 604800  # 7 days
assert SOURCE_TIERS['web']['cache_ttl'] == 86400           # 24 hours
print(f'PASS: AG-Retrieve  top_k={ag_r.top_k_retrieve}->{ag_r.top_k_final}, {len(SOURCE_TIERS)} source tiers, TTLs OK')

# 2. AG-Execute
from antigravity.ag_execute import AGExecute, SANDBOX_IMAGES
ag_e = AGExecute(use_gvisor=True, fallback_to_subprocess=True)
assert ag_e.TIMEOUT_SECONDS == 5
assert ag_e.MEMORY_LIMIT == "256m"
assert 'python' in SANDBOX_IMAGES and '3.12' in SANDBOX_IMAGES['python']
assert 'go'     in SANDBOX_IMAGES and '1.22' in SANDBOX_IMAGES['go']
assert 'java'   in SANDBOX_IMAGES and '21'   in SANDBOX_IMAGES['java']
assert 'rust'   in SANDBOX_IMAGES
assert 'swift'  in SANDBOX_IMAGES and '5.10' in SANDBOX_IMAGES['swift']
result = ag_e.run('print(42)', 'python')
assert result.stdout.strip() == '42'
print(f'PASS: AG-Execute   timeout={ag_e.TIMEOUT_SECONDS}s mem={ag_e.MEMORY_LIMIT} sandbox->42')

# 3. AG-Verify
from antigravity.ag_verify import AGVerify, CONFIDENCE_THRESHOLD, W_CITATION, W_EXECUTION, W_CONSISTENCY
assert CONFIDENCE_THRESHOLD == 0.75
assert abs(W_CITATION + W_EXECUTION + W_CONSISTENCY - 1.0) < 1e-9
ag_v = AGVerify()
v = ag_v.verify('generic uncited response', [{'text':'docs','url':'https://python.org'}], None, False)
print(f'PASS: AG-Verify    threshold={CONFIDENCE_THRESHOLD}, weights=({W_CITATION},{W_EXECUTION},{W_CONSISTENCY})')
print(f'      sample score={v.score.composite:.2f}, intercepted={v.intercepted}')

# 4. AG-Cache
from antigravity.ag_cache import AGCache, SIMILARITY_THRESHOLD, DOC_TTL, DEFAULT_TTL
assert SIMILARITY_THRESHOLD == 0.92
assert DOC_TTL == 604800    # 7 days
assert DEFAULT_TTL == 86400  # 24 hours
ag_c = AGCache()
ttl_github = ag_c._ttl_for_tier('github')
assert ttl_github == 3600    # 1 hour per spec
print(f'PASS: AG-Cache     sim_threshold={SIMILARITY_THRESHOLD}, github_ttl={ttl_github}s, doc_ttl={DOC_TTL}s')

# 5. AG-Update
from antigravity.ag_update import AGUpdate
ag_u = AGUpdate()
assert ag_u.confidence_threshold == 0.75
assert ag_u.min_examples_for_lora == 500
print(f'PASS: AG-Update    threshold={ag_u.confidence_threshold}, min_lora_examples={ag_u.min_examples_for_lora}')

# 6. Pipeline
from antigravity.pipeline import AntigravityPipeline
dev = AntigravityPipeline.dev()
st = dev.status()
assert st['retrieval'] == True
assert st['sandbox'] == True
assert st['verify'] == True
print(f'PASS: Pipeline     retrieval={st["retrieval"]}, sandbox={st["sandbox"]}, verify={st["verify"]}')

# 7. Package exports
from antigravity import (AGRetrieve, AGExecute, AGVerify, AGCache,
                          AGUpdate, AntigravityPipeline, build_rag_prompt)
print('PASS: Package __init__ exports OK')

print()
print('=== ALL ANTIGRAVITY CHECKS PASSED ===')
