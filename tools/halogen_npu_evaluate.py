#!/usr/bin/env python3
"""Bounded NPU ingestion bursts and full-corpus BEIR SciFact retrieval evaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request
import zipfile

DATASET_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
DATASET_SHA256 = '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165'
INSTRUCTION = 'Given a scientific claim, retrieve documents that support or refute the claim'
EMBEDDER = 'qwen3-embedding-0.6b'
RERANKER = 'qwen3-reranker-0.6b'


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)


def request(base, path, payload=None):
    req = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def dataset(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError('SciFact archive does not match the pinned checksum')
    with zipfile.ZipFile(path) as archive:
        corpus = [json.loads(line) for line in archive.read('scifact/corpus.jsonl').decode().splitlines()]
        queries = {q['_id']: q['text'] for q in map(json.loads, archive.read('scifact/queries.jsonl').decode().splitlines())}
        qrels = defaultdict(dict)
        for row in csv.DictReader(io.StringIO(archive.read('scifact/qrels/test.tsv').decode()), delimiter='\t'):
            if int(row['score']) > 0:
                qrels[row['query-id']][row['corpus-id']] = int(row['score'])
    corpus.sort(key=lambda d: d['_id'])
    return corpus, queries, dict(qrels)


def validate_vectors(response, count):
    rows = sorted(response['data'], key=lambda row: row['index'])
    if [r['index'] for r in rows] != list(range(count)):
        raise ValueError('missing or duplicated embedding indices')
    vectors = [r['embedding'] for r in rows]
    for vector in vectors:
        if len(vector) != 1024 or not all(math.isfinite(x) for x in vector):
            raise ValueError('invalid embedding vector')
        if abs(sum(x*x for x in vector) - 1) > 2e-4:
            raise ValueError('embedding is not normalized')
    return vectors


def metrics(ranking, relevant):
    if len(ranking) != len(set(ranking)):
        raise ValueError('duplicate document IDs in ranking')
    result = {}
    for k in (1, 5, 10, 100):
        gains = [relevant.get(identifier, 0) for identifier in ranking[:k]]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        result[f'ndcg@{k}'] = dcg / idcg if idcg else 0
        result[f'recall@{k}'] = sum(g > 0 for g in gains) / len(relevant)
    result['mrr@10'] = next((1 / (i + 1) for i, identifier in enumerate(ranking[:10]) if identifier in relevant), 0)
    return result


def burst(args, record, corpus):
    # Match #127's 40 concurrent requests of ten inputs without client retries.
    repeated = ['lorem ipsum ' * 200] + list('bcdefghij')
    reference = validate_vectors(request(args.base_url, '/v1/embeddings', {'model': EMBEDDER, 'input': repeated}), 10)
    scenarios = [('issue-127-40x10', 40, [repeated] * 40)]
    # Treat each of 188 real abstracts as a file, split into <=120-word chunks.
    files = []
    for doc in corpus[:188]:
        words = (doc['title'] + '\n' + doc['text']).split()
        files.append([' '.join(words[i:i+120]) for i in range(0, len(words), 120)])
    scenarios += [('scifact-188-files-concurrency-12', 12, files), ('scifact-188-files-concurrency-40', 40, files)]
    record['scenarios'] = []
    for name, workers, inputs in scenarios:
        start = time.monotonic()
        def one(index):
            begin = time.monotonic()
            row = {'request': index, 'inputs': len(inputs[index])}
            try:
                result = request(args.base_url, '/v1/embeddings', {'model': EMBEDDER, 'input': inputs[index]})
                vectors = validate_vectors(result, len(inputs[index]))
                row.update(status=200, valid=True, total_tokens=result['usage']['total_tokens'])
                if name == 'issue-127-40x10':
                    delta = max(abs(a-b) for before, after in zip(reference, vectors) for a,b in zip(before, after))
                    row.update(max_abs_reference_difference=delta, valid=delta < 1e-5)
            except urllib.error.HTTPError as exc:
                row.update(status=exc.code, valid=False, error=exc.read().decode(), retry_after=exc.headers.get('Retry-After'))
            except Exception as exc:
                row.update(status=None, valid=False, error=repr(exc))
            row['seconds'] = time.monotonic() - begin
            return row
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, range(len(inputs))))
        elapsed = time.monotonic() - start
        times = sorted(r['seconds'] for r in rows)
        entry = {'name': name, 'workers': workers, 'requests': len(rows), 'inputs': sum(r['inputs'] for r in rows),
                 'wall_seconds': elapsed, 'status_counts': dict(Counter(str(r['status']) for r in rows)),
                 'latency_median_seconds': statistics.median(times), 'latency_p95_seconds': times[math.ceil(.95*len(times))-1],
                 'latency_max_seconds': max(times), 'all_passed': all(r['valid'] for r in rows), 'results': rows}
        record['scenarios'].append(entry)
        save(args.output, record)
        print(name, entry['status_counts'], 'passed', entry['all_passed'], f'{elapsed:.1f}s', flush=True)
        if not entry['all_passed']:
            raise RuntimeError('burst failed; stopping before higher load')


def quality(args, record, corpus, queries, qrels):
    import numpy as np
    docs = [d['title'] + '\n' + d['text'] for d in corpus]
    ids = [d['_id'] for d in corpus]
    qids = sorted(qrels)
    record['method'] = {'dataset': 'BEIR SciFact test', 'corpus_count': len(docs), 'queries': len(qids),
                        'documents': 'title + newline + abstract, no truncation', 'query_instruction': INSTRUCTION,
                        'embedding_dimensions': 1024, 'similarity': 'cosine, normalized vectors', 'batch_size': 8,
                        'rerank_candidates': 10, 'rerank_instruction': INSTRUCTION,
                        'lexical_baseline': 'BM25 k1=1.2 b=0.75; Unicode word tokens, lowercase, no stemming/stopword removal',
                        'limits': 'One dataset; no BF16 reference or general MTEB qualification. Reranking changes only top 10; recall@10 cannot increase.'}
    manifest = Path(__file__).resolve().parents[1] / 'lib/halo_ai/halogen_npu_models.json'
    fingerprint = hashlib.sha256(json.dumps({'dataset': DATASET_SHA256, 'model': EMBEDDER, 'dimensions': 1024,
        'version': record['health_before']['version'], 'npu_manifest': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'documents': record['method']['documents']}, sort_keys=True).encode()).hexdigest()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.cache_dir / (fingerprint + '.npy')
    meta_path = args.cache_dir / (fingerprint + '.json')
    matrix = np.zeros((len(docs), 1024), dtype=np.float32)
    completed = 0
    if meta_path.exists() and vector_path.exists():
        meta = json.loads(meta_path.read_text())
        completed = meta['completed']
        matrix = np.load(vector_path, allow_pickle=False)
        if matrix.shape != (len(docs), 1024) or meta['fingerprint'] != fingerprint:
            raise ValueError('embedding cache shape/provenance mismatch')
    record['cache'] = {'fingerprint': fingerprint, 'reused_documents': completed}
    stage_start = time.monotonic()
    for offset in range(completed, len(docs), 8):
        texts = docs[offset:offset+8]
        response = request(args.base_url, '/v1/embeddings', {'model': EMBEDDER, 'input': texts})
        matrix[offset:offset+len(texts)] = validate_vectors(response, len(texts))
        completed = offset + len(texts)
        if completed % 64 == 0 or completed == len(docs):
            temporary = vector_path.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.save(stream, matrix, allow_pickle=False)
            temporary.replace(vector_path)
            save(meta_path, {'fingerprint': fingerprint, 'completed': completed})
            record['corpus_embeddings_completed'] = completed
            save(args.output, record)
            print(f'Embedded {completed}/{len(docs)} documents ({time.monotonic()-stage_start:.1f}s this run)', flush=True)
    record['corpus_embedding_seconds_this_run'] = time.monotonic() - stage_start
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= norms[:, None]
    qvectors = []
    for offset in range(0, len(qids), 8):
        texts = ['Instruct: ' + INSTRUCTION + '\nQuery:' + queries[q] for q in qids[offset:offset+8]]
        qvectors.extend(validate_vectors(request(args.base_url, '/v1/embeddings', {'model': EMBEDDER, 'input': texts}), len(texts)))
    qmatrix = np.asarray(qvectors, dtype=np.float32)
    qmatrix /= np.linalg.norm(qmatrix, axis=1)[:, None]
    similarities = qmatrix @ matrix.T
    # An inexpensive independent lexical baseline, evaluated on identical qrels.
    posting = defaultdict(list)
    lengths = []
    for index, document in enumerate(docs):
        terms = Counter(re.findall(r'\w+', document.lower()))
        lengths.append(sum(terms.values()))
        for term, count in terms.items():
            posting[term].append((index, count))
    average = statistics.mean(lengths)
    record['queries'] = []
    start = time.monotonic()
    for qi, qid in enumerate(qids):
        indices = np.argsort(-similarities[qi], kind='stable')[:100].tolist()
        bm25 = np.zeros(len(docs), dtype=np.float64)
        for term in set(re.findall(r'\w+', queries[qid].lower())):
            entries = posting.get(term, [])
            idf = math.log(1 + (len(docs) - len(entries) + .5) / (len(entries) + .5))
            for index, tf in entries:
                bm25[index] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[index] / average))
        lexical = np.argsort(-bm25, kind='stable')[:100].tolist()
        begin = time.monotonic()
        response = request(args.base_url, '/v1/rerank', {'model': RERANKER, 'query': queries[qid],
            'documents': [docs[i] for i in indices[:10]], 'instruction': INSTRUCTION})
        ranked = response['results']
        if sorted(r['index'] for r in ranked) != list(range(10)) or not all(math.isfinite(r['relevance_score']) for r in ranked):
            raise ValueError('invalid rerank permutation/scores')
        reranked = [indices[r['index']] for r in ranked] + indices[10:]
        rankings = {'embedding': [ids[i] for i in indices], 'embedding_reranked': [ids[i] for i in reranked], 'bm25': [ids[i] for i in lexical]}
        record['queries'].append({'id': qid, 'rankings': rankings, 'metrics': {name: metrics(ranking, qrels[qid]) for name, ranking in rankings.items()},
                                  'rerank_seconds': time.monotonic()-begin, 'rerank_scores': [r['relevance_score'] for r in ranked], 'rerank_usage': response.get('usage')})
        if len(record['queries']) % 10 == 0:
            save(args.output, record)
            print(f'Reranked {len(record["queries"])}/{len(qids)} queries ({time.monotonic()-start:.1f}s)', flush=True)
    record['summary'] = {name: {metric: statistics.mean(row['metrics'][name][metric] for row in record['queries'])
        for metric in record['queries'][0]['metrics'][name]} for name in rankings}
    differences = [row['metrics']['embedding_reranked']['ndcg@10'] - row['metrics']['embedding']['ndcg@10'] for row in record['queries']]
    rng = np.random.default_rng(1)
    means = np.mean(rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1)
    record['paired_ndcg10_change'] = {'mean': statistics.mean(differences), 'bootstrap_95_percent_interval': np.quantile(means, [.025, .975]).tolist(),
        'improved_queries': sum(d > 1e-12 for d in differences), 'worsened_queries': sum(d < -1e-12 for d in differences), 'bootstrap_seed': 1, 'bootstrap_samples': 5000}
    print(json.dumps(record['summary'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['burst', 'quality'])
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--dataset', type=Path, default=Path('/var/cache/halo-ai/benchmarks/scifact/scifact.zip'))
    parser.add_argument('--cache-dir', type=Path, default=Path('/var/cache/halo-ai/benchmarks/scifact/embeddings'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus, queries, qrels = dataset(args.dataset)
    health = request(args.base_url, '/health')
    if health.get('model') != 'halogen-qwen3.8-flash-next-v2' or health.get('npu', {}).get('status') != 'ok' or health.get('version', {}).get('api') != '0.16.2':
        parser.error('requires healthy Halogen 0.16.2 v2 with NPU services')
    record = {'schema_version': 1, 'mode': args.mode, 'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'dataset_url': DATASET_URL, 'dataset_sha256': DATASET_SHA256, 'health_before': health, 'completed': False,
              'request_timeout_seconds': 180, 'client_retries': 0}
    save(args.output, record)
    try:
        if args.mode == 'burst':
            burst(args, record, corpus)
        else:
            quality(args, record, corpus, queries, qrels)
        record['health_after'] = request(args.base_url, '/health')
        record['completed'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        if isinstance(exc, urllib.error.HTTPError):
            record['error_body'] = exc.read().decode()
        raise
    finally:
        record['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(args.output, record)


if __name__ == '__main__':
    main()
