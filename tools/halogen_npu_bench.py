#!/usr/bin/env python3
"""Measure isolated NPU HTTP throughput around the short-input batching boundary."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sentence = 'Paris is the capital of France. The city has museums, parks, bridges, and a river. '
    record = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'method': 'Each endpoint alone: one warmup then three measured requests. Run without other inference clients. GPU model remains loaded but idle. HTTP wall latency includes endpoint overhead. Synthetic repeated texts, not upstream benchmark data.', 'sentence': sentence, 'measurements': []}
    cases = [('embeddings', 25, 8), ('embeddings', 26, 8), ('rerank', 21, 8), ('rerank', 26, 8), ('embeddings', 25, 1)]
    for endpoint, repetitions, batch in cases:
        payload = {'model': 'qwen3-embedding-0.6b' if endpoint == 'embeddings' else 'qwen3-reranker-0.6b'}
        texts = [sentence * repetitions] * batch
        payload.update({'input': texts} if endpoint == 'embeddings' else {'query': 'What is the capital of France?', 'documents': texts})
        samples = []
        for index in range(4):
            req = urllib.request.Request(args.base_url + '/v1/' + endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.load(response)
            elapsed = time.monotonic() - start
            tokens = result['usage']['total_tokens']
            if index:
                samples.append({'seconds': elapsed, 'total_tokens': tokens, 'tokens_per_input': tokens / batch, 'tokens_per_second': tokens / elapsed, 'inputs_per_second': batch / elapsed})
        entry = {'endpoint': endpoint, 'sentence_repetitions': repetitions, 'batch_size': batch, 'samples': samples,
                 'median_seconds': statistics.median(s['seconds'] for s in samples),
                 'median_tokens_per_second': statistics.median(s['tokens_per_second'] for s in samples),
                 'median_inputs_per_second': statistics.median(s['inputs_per_second'] for s in samples)}
        record['measurements'].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(endpoint, samples[0]['tokens_per_input'], 'tokens/input', batch, 'inputs:', round(entry['median_tokens_per_second'], 1), 'tok/s;', round(entry['median_inputs_per_second'], 2), 'inputs/s', flush=True)


if __name__ == '__main__':
    main()
