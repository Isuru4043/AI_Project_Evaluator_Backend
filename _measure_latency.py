"""Measure raw per-model LLM latency. Delete after."""
import os, time
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Evaluator_Backend.settings')
import django
django.setup()

from viva_evaluator.services.llm_service import llm_call, MODEL_REGISTRY

prompt = ("You are an academic viva examiner. Generate one short question about "
          "a student's use of AES-256-GCM encryption in a zero-trust file transfer "
          "system. Respond in JSON: {\"question_text\": \"...\"}")

for model in ['reasoning', 'fast']:
    model_id = MODEL_REGISTRY.get(model)
    times = []
    for i in range(3):
        t0 = time.time()
        llm_call(prompt, model=model, expect_json=True, max_retries=0, fallback={})
        dt = time.time() - t0
        times.append(dt)
        print(f'  {model} ({model_id}) call {i+1}: {dt:.1f}s')
    avg = sum(times) / len(times)
    print(f'  => {model} avg = {avg:.1f}s\n')

print('Done.')
