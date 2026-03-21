"""Modelworks component verification script."""
import sys
sys.path.insert(0, 'd:/MODEL')

print('=== Modelworks Component Import Verification ===')

# 1. Model-Retrieve
from Modelworks.retrieve import ModelRetrieve, SOURCE_TIERS
model_r = ModelRetrieve()
assert model_r.top_k_retrieve == 10 and model_r.top_k_final == 3
required_tiers = ['github', 'stackoverflow', 'mdn', 'pypi_npm', 'lang_refs', 'arxiv', 'web']
assert all(t in SOURCE_TIERS for t in required_tiers)
# TTL checks from spec
assert SOURCE_TIERS['github']['cache_ttl'] == 3600         # 1 hour
assert SOURCE_TIERS['stackoverflow']['cache_ttl'] == 604800  # 7 days
assert SOURCE_TIERS['web']['cache_ttl'] == 86400           # 24 hours
print(f'PASS: Model-Retrieve  top_k={model_r.top_k_retrieve}->{model_r.top_k_final}, {len(SOURCE_TIERS)} source tiers, TTLs OK')

# 2. Model-Execute
from Modelworks.execute import ModelExecute, SANDBOX_IMAGES
model_e = ModelExecute(use_gvisor=True, fallback_to_subprocess=True)
assert model_e.TIMEOUT_SECONDS == 5
assert model_e.MEMORY_LIMIT == "256m"
assert 'python' in SANDBOX_IMAGES and '3.12' in SANDBOX_IMAGES['python']
assert 'go'     in SANDBOX_IMAGES and '1.22' in SANDBOX_IMAGES['go']
assert 'java'   in SANDBOX_IMAGES and '21'   in SANDBOX_IMAGES['java']
assert 'rust'   in SANDBOX_IMAGES
assert 'swift'  in SANDBOX_IMAGES and '5.10' in SANDBOX_IMAGES['swift']
result = model_e.run('print(42)', 'python')
assert result.stdout.strip() == '42'
print(f'PASS: Model-Execute   timeout={model_e.TIMEOUT_SECONDS}s mem={model_e.MEMORY_LIMIT} sandbox->42')

# 3. Model-Verify
from Modelworks.verify import ModelVerify, CONFIDENCE_THRESHOLD, W_CITATION, W_EXECUTION, W_CONSISTENCY
assert CONFIDENCE_THRESHOLD == 0.75
assert abs(W_CITATION + W_EXECUTION + W_CONSISTENCY - 1.0) < 1e-9
model_v = ModelVerify()
v = model_v.verify('generic uncited response', [{'text':'docs','url':'https://python.org'}], None, False)
print(f'PASS: Model-Verify    threshold={CONFIDENCE_THRESHOLD}, weights=({W_CITATION},{W_EXECUTION},{W_CONSISTENCY})')
print(f'      sample score={v.score.composite:.2f}, intercepted={v.intercepted}')

# 4. Model-Cache
from Modelworks.cache import ModelCache, SIMILARITY_THRESHOLD, DOC_TTL, DEFAULT_TTL
assert SIMILARITY_THRESHOLD == 0.92
assert DOC_TTL == 604800    # 7 days
assert DEFAULT_TTL == 86400  # 24 hours
model_c = ModelCache()
ttl_github = model_c._ttl_for_tier('github')
assert ttl_github == 3600    # 1 hour per spec
print(f'PASS: Model-Cache     sim_threshold={SIMILARITY_THRESHOLD}, github_ttl={ttl_github}s, doc_ttl={DOC_TTL}s')

# 5. Model-Update
from Modelworks.update import ModelUpdate
model_u = ModelUpdate()
assert model_u.confidence_threshold == 0.75
assert model_u.min_examples_for_lora == 500
print(f'PASS: Model-Update    threshold={model_u.confidence_threshold}, min_lora_examples={model_u.min_examples_for_lora}')

# 6. Pipeline
from Modelworks.pipeline import ModelPipeline
dev = ModelPipeline.dev()
st = dev.status()
assert st['retrieval'] == True
assert st['sandbox'] == True
assert st['verify'] == True
print(f'PASS: Pipeline     retrieval={st["retrieval"]}, sandbox={st["sandbox"]}, verify={st["verify"]}')

# 7. Package exports
from Modelworks import (ModelRetrieve, ModelExecute, ModelVerify, ModelCache,
                          ModelUpdate, ModelPipeline, build_rag_prompt)
print('PASS: Package __init__ exports OK')

print()
print('=== ALL MODELWORKS CHECKS PASSED ===')
