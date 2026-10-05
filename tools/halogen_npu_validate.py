#!/usr/bin/env python3
"""Exercise NPU APIs, burst handling, and bounded concurrent GPU/NPU serving."""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3, help='Paired timing rounds; 0 runs correctness and concurrent reference checks only')
    args = parser.parse_args()
    if not 0 <= args.rounds <= 10:
        parser.error('--rounds must be between 0 and 10')
    embedder, reranker = 'qwen3-embedding-0.6b', 'qwen3-reranker-0.6b'

    def call(path, payload=None):
        req = urllib.request.Request(args.base_url + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    health = call('/health')
    models = call('/v1/models')
    if not {embedder, reranker, health['model']} <= {m['id'] for m in models['data']}:
        parser.error('start qwen3.8-halogen-npu with both NPU models')
    record = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'health_before': health, 'models': models, 'method': 'Synthetic API checks plus alternating paired GPU decode probes with the NPU idle or continuously embedding/reranking. Same running process, cache mode 2, greedy and thinking off. Bounded stress observations; not model-quality, gaming, or long-term stability qualification.', 'checks': [], 'probes': []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')

    def check(name, function):
        start = time.monotonic()
        try:
            evidence = function()
            result = {'name': name, 'passed': True, 'evidence': evidence}
        except Exception as exc:
            result = {'name': name, 'passed': False, 'error': str(exc)}
            if hasattr(exc, 'read'):
                result['error_body'] = exc.read().decode()
        result['wall_seconds'] = time.monotonic() - start
        record['checks'].append(result)
        save()
        print(name, result['passed'], flush=True)

    def embeddings(texts, **extra):
        return call('/v1/embeddings', {'model': embedder, 'input': texts, **extra})

    def vectors(response, count, dimensions=1024):
        data = sorted(response['data'], key=lambda x: x['index'])
        assert [d['index'] for d in data] == list(range(count)), response
        values = [d['embedding'] for d in data]
        for vector in values:
            assert len(vector) == dimensions and all(math.isfinite(x) for x in vector)
            assert abs(math.sqrt(sum(x*x for x in vector)) - 1) < 1e-4
        return values

    texts = ['Instruct: Given a question, retrieve passages that answer it\nQuery:What is the capital of France?', 'Bananas are yellow fruit.', 'Paris is the capital of France.']
    reference = {}

    def embedding_check():
        batch = vectors(embeddings(texts), 3)
        single = vectors(embeddings([texts[2]]), 1)[0]
        difference = max(abs(a-b) for a,b in zip(batch[2], single))
        assert difference < 1e-5, difference
        scores = [sum(a*b for a,b in zip(batch[0], vector)) for vector in batch[1:]]
        assert scores[1] > scores[0], scores
        if 'embedding' in reference:
            assert max(abs(a-b) for a,b in zip(batch[2], reference['embedding'])) < 1e-5
        reference['embedding'] = batch[2]
        return {'dimensions': 1024, 'unit_norm': True, 'batch_single_max_abs_difference': difference, 'retrieval_scores_fruit_then_paris': scores, 'vector_sha256': hashlib.sha256(json.dumps(single).encode()).hexdigest()}
    check('embeddings-shape-normalization-batching-retrieval', embedding_check)

    def formats_check():
        small = vectors(embeddings([texts[2]], dimensions=128), 1, 128)[0]
        full = reference['embedding']
        norm = math.sqrt(sum(x*x for x in full[:128]))
        difference = max(abs(a-b/norm) for a,b in zip(small, full[:128]))
        assert difference < 1e-5, difference
        encoded = embeddings([texts[2]], encoding_format='base64')['data'][0]['embedding']
        binary = struct.unpack('<1024f', base64.b64decode(encoded))
        binary_difference = max(abs(a-b) for a,b in zip(binary, full))
        assert binary_difference < 1e-5, binary_difference
        return {'reduced_dimensions': 128, 'prefix_renormalization_max_difference': difference, 'base64_float_max_difference': binary_difference}
    check('embedding-dimensions-and-base64', formats_check)

    documents = ['Bananas are yellow fruit.', 'Paris is the capital of France.', 'Tokyo is the capital of Japan.']

    def rerank(docs, **extra):
        return call('/v1/rerank', {'model': reranker, 'query': 'What is the capital of France?', 'documents': docs, **extra})

    def validate_ranking(response, count):
        results = response['results']
        assert len(results) == count, response
        assert len({r['index'] for r in results}) == count, response
        scores = [r['relevance_score'] for r in results]
        assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores), scores
        assert scores == sorted(scores, reverse=True), scores
        return results

    def reranking_check():
        results = validate_ranking(rerank(documents), 3)
        assert results[0]['index'] == 1, results
        top = validate_ranking(rerank([{'text': d} for d in documents], top_n=1, return_documents=True), 1)
        assert top[0]['index'] == 1 and top[0]['document']['text'] == documents[1], top
        by_index = {r['index']: r['relevance_score'] for r in results}
        if 'rerank' in reference:
            assert max(abs(score-reference['rerank'][i]) for i,score in by_index.items()) < 1e-5
        reference['rerank'] = by_index
        return {'all_results': results, 'top_document': top}
    check('rerank-order-top-n-and-document-objects', reranking_check)

    def limits_check():
        errors = []
        for path, body in [('/v1/embeddings', {'model': embedder, 'input': 'hello ' * 6000}), ('/v1/rerank', {'model': reranker, 'query': 'capital', 'documents': ['hello ' * 6000]})]:
            try:
                call(path, body)
                raise AssertionError(f'{path} accepted an oversized input')
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                assert exc.code == 400, (exc.code, body)
                errors.append({'route': path, 'status': exc.code, 'body': body})
        return errors
    check('oversized-inputs-return-400', limits_check)

    def burst_check():
        def one(_):
            start = time.monotonic()
            vector = vectors(embeddings([texts[2]] * 8), 8)
            delta = max(abs(a-b) for v in vector for a,b in zip(v, reference['embedding']))
            assert delta < 1e-5, delta
            return time.monotonic() - start
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            durations = list(pool.map(one, range(16)))
        return {'requests': len(durations), 'inputs_per_request': 8, 'median_seconds': statistics.median(durations), 'max_seconds': max(durations), 'all_vectors_match_single': True}
    check('16-request-embedding-burst', burst_check)
    # Stop here if basic correctness failed; avoid obscuring the cause with load.
    if any(not c['passed'] for c in record['checks']):
        return 1

    load_texts = [('Paris is the capital of France. The city has museums, parks, bridges, and a river. ' * 26) for _ in range(8)]
    def gpu_probe():
        start = time.monotonic()
        response = call('/v1/chat/completions', {'model': health['model'], 'messages': [{'role': 'user', 'content': 'Write a Python LinkedList class with push, pop, reverse, __len__, and __iter__. Include docstrings and type hints.'}], 'max_tokens': 512, 'temperature': 0, 'seed': 1, 'reasoning_effort': 'none'})
        content = response['choices'][0]['message']['content']
        assert content and response['usage']['completion_tokens'] > 0, response
        return {'wall_seconds': time.monotonic() - start, 'sha256': hashlib.sha256(content.encode()).hexdigest(), 'finish_reason': response['choices'][0]['finish_reason'], 'usage': response['usage'], 'timings': response['timings']}

    for round_index in range(args.rounds):
        for mode in (['idle', 'busy'] if round_index % 2 == 0 else ['busy', 'idle']):
            entry = {'round': round_index + 1, 'npu': mode}
            stop = threading.Event()
            started = [threading.Event(), threading.Event()]
            def load(kind):
                count = tokens = 0
                elapsed = []
                while not stop.is_set():
                    started[kind].set()
                    begin = time.monotonic()
                    if kind == 0:
                        result = embeddings(load_texts)
                        vectors(result, 8)
                        tokens += result.get('usage', {}).get('total_tokens', 0)
                    else:
                        validate_ranking(rerank(load_texts), 8)
                    elapsed.append(time.monotonic() - begin)
                    count += 1
                return {'requests': count, 'inputs': count * 8, 'reported_tokens': tokens, 'total_request_seconds': sum(elapsed), 'median_request_seconds': statistics.median(elapsed) if elapsed else None}
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(load, kind) for kind in range(2)] if mode == 'busy' else []
                    if jobs:
                        assert all(event.wait(10) for event in started)
                    try:
                        entry['gpu'] = gpu_probe()
                    finally:
                        stop.set()
                    if jobs:
                        entry['embedding_load'], entry['rerank_load'] = [job.result() for job in jobs]
                        assert entry['embedding_load']['requests'] and entry['rerank_load']['requests']
                entry['passed'] = True
            except Exception as exc:
                entry.update(passed=False, error=str(exc))
            record['probes'].append(entry)
            save()
            print('GPU probe', round_index + 1, mode, entry.get('gpu', {}).get('timings', {}), 'passed', entry['passed'], flush=True)
            if not entry['passed']:
                return 1
    def concurrent_reference_check():
        # Compare actual values under load, not merely finite/normalized shapes.
        done = threading.Event()
        started = threading.Event()
        def generate():
            started.set()
            try:
                return gpu_probe()
            finally:
                done.set()
        counts = {'embedding': 0, 'rerank': 0}
        differences = {'embedding': 0.0, 'rerank': 0.0}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            gpu = pool.submit(generate)
            assert started.wait(10)
            while not done.is_set():
                batch = vectors(embeddings([texts[2]] * 8), 8)
                delta = max(abs(a-b) for vector in batch for a,b in zip(vector, reference['embedding']))
                assert delta < 1e-5, delta
                differences['embedding'] = max(differences['embedding'], delta)
                counts['embedding'] += 1
                ranking = validate_ranking(rerank(documents), 3)
                delta = max(abs(r['relevance_score']-reference['rerank'][r['index']]) for r in ranking)
                assert delta < 1e-5, delta
                differences['rerank'] = max(differences['rerank'], delta)
                counts['rerank'] += 1
            result = gpu.result()
        assert all(counts.values()), counts
        return {'requests': counts, 'max_absolute_differences_from_idle': differences, 'gpu': result}
    check('concurrent-gpu-npu-reference-values', concurrent_reference_check)
    # Check correctness again after shared GPU/NPU load.
    check('post-load-embedding-repeat', embedding_check)
    check('post-load-rerank-repeat', reranking_check)
    record['health_after'] = call('/health')
    record['summary'] = {mode + '_median_gpu_decode_tps': statistics.median(p['gpu']['timings']['predicted_per_second'] for p in record['probes'] if p['npu'] == mode) for mode in ['idle', 'busy']} if record['probes'] else {}
    record['summary']['all_checks_passed'] = all(c['passed'] for c in record['checks']) and all(p['passed'] for p in record['probes'])
    save()
    print(json.dumps(record['summary'], indent=2), flush=True)
    return int(not record['summary']['all_checks_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
