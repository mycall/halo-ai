# Speech translation

[Documentation index](README.md) · [Project README](../README.md)

Install the host tools first using [Installation](installation.md). The speech service shares the managed runtime lifecycle described in [Operations](operations.md).

## Seamless speech translation

The speech service follows AMD's current gfx1151 ROCm playbook and is independent
of Lemonade and the LLM runtimes. Build its cached image layers, download only
the pinned safetensors/processor subset, and verify the local bytes:

```bash
halo-ai install speech
halo-ai models download seamless-m4t-v2-large
halo-ai models verify seamless-m4t-v2-large --full
```

The model is stored directly at
`/srv/halo-ai/models/facebook/seamless-m4t-v2-large`; there is no intermediate
`huggingface/` directory. Downloads resume partial files and reuse complete
ones. The 7.1 GB local image retains the AMD ROCm/PyTorch wheels in Podman's
layer cache, while the model remains external and read-only at runtime.

```bash
halo-ai start seamless-m4t-v2-large-speech --switch
halo-ai test seamless-m4t-v2-large-speech
halo-ai status
```

Open `http://127.0.0.1:7860/` for the local Gradio UI. Automation can use
`GET /healthz` and multipart `POST /api/v1/translate` on the same loopback port.
The built-in test sends AMD's hash-pinned sample to Spanish (`spa`) and validates
that the response is a non-empty WAV. `halo-ai stop` includes this service;
normal uninstall still preserves the external model root.

Query the available three-letter language codes and translate a local audio file
to Ukrainian with:

```bash
curl --fail http://127.0.0.1:7860/healthz | jq '.target_languages'

curl --fail-with-body \
  http://127.0.0.1:7860/api/v1/translate \
  -F 'audio=@/path/to/input.wav' \
  -F 'target_lang=ukr' \
  --output translated-uk.wav

file translated-uk.wav
halo-ai stop
```

The service accepts common audio formats supported by libsndfile, resamples the
input to 16 kHz when necessary, and returns a 16 kHz PCM WAV. Its response
headers include `X-Halo-AI-Inference-Seconds` and
`X-Halo-AI-Target-Language`; add `--dump-header -` to the translation command
when those values are useful. The server derives its language list from the
loaded checkpoint and exposes all 36 languages supported for both speech input
and speech output; the API and Gradio UI use the same list.
