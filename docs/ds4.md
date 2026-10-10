# DeepSeek and DS4

[Documentation index](README.md) · [Project README](../README.md)

Install the host tools first using [Installation](installation.md). See
[Profiles](profiles.md) for artifact acquisition and [Operations](operations.md)
for switching and stopping services. The commands below assume the selected
model artifacts have already been acquired.

## DS4 disk KV cache

The `ds4-deepseek-v4-flash-hybrid` profile is the uncached 32K control. Use the separate
cache profile when coding agents or other clients repeatedly send a long shared
prefix:

```bash
halo-ai install ds4
halo-ai start ds4-deepseek-v4-flash-hybrid-kv --switch
halo-ai test ds4-deepseek-v4-flash-hybrid-kv
```

The cache profile keeps the proven 32K context and 2,048-token prefill chunk,
adds an 8 GiB budgeted cache at `/var/cache/halo-ai/ds4-kv`, and rejects cache
entries from a different quantization. Cache files survive `stop`, restart, and
container recreation; they do not reduce the memory required by an active
context. The first prompt must still be processed in full before a reusable
entry exists.

## DSpark and Think Max

DSpark requires the pinned b0001 runtime with the ROCm proposal/verification
kernels. The enabled profiles remain experimental: nonzero accepted drafts
confirm working speculation, but a bounded decode probe was slower than the
target-only path. Speculation is not a guaranteed speed improvement.

| Profile suffix | Context | Use |
| --- | ---: | --- |
| `hybrid-dspark-16k` | 16,384 | Low-memory diagnostic rollback |
| `hybrid-dspark-128k` | 131,072 | More host-memory headroom |
| `hybrid-dspark-256k` | 262,144 | Upper work profile |
| `hybrid-dspark-384k-think-max` | 393,216 | Opt-in Think Max capacity; alias `ds4` |

Prefix these suffixes with `ds4-deepseek-v4-flash-`. Profiles above 16K use the
persistent 8 GiB disk-KV policy. The target plus support file occupies 96.47 GiB;
model architecture support for 1M context does not establish host capacity, and
no 1M profile is exposed.

The 128K, 256K, and 384K tiers passed allocation and short-request checks, not
full-length prompt qualification. Recorded GTT use was 104.38, 106.30, and
108.23 GiB respectively, with only 8.40 GiB host memory available at 384K.
For ordinary work, choose 256K or 128K when other host workloads need headroom.

`ds4` resolves to the full 384K Think Max profile. Clients still need to request
max reasoning; the profile supplies capacity, and the preset supplies the
request policy:

```bash
halo-ai start ds4 --switch
halo-ai test ds4 --preset deepseek-v4-think-max
```

## API usage

DS4 exposes an OpenAI-compatible API on loopback. Inspect the loaded model and
send a non-thinking chat request with:

```bash
curl --fail http://127.0.0.1:8000/v1/models | jq

curl --fail-with-body \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Explain why a persistent KV cache helps repeated long prompts."}
    ],
    "reasoning_effort": "none",
    "max_tokens": 128,
    "stream": false
  }' | jq '.choices[0].message'

halo-ai stop
```
