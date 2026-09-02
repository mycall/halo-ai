Current preparation state:

1. The four Unsloth `UD-Q4_K_XL` shards are present under `/srv`, total
   111,334,654,784 bytes, and their local SHA-256 values match Hugging Face
   revision `38bb39ee97821de2c9009abb7e93950eec396e66`.
2. Model `qwen3.8-flash-next-ud-q4-k-xl` and experimental 32K profile
   `qwen3.8-flash-next-ud-q4-k-xl-llamacpp` are cataloged. Nested Hugging Face
   source paths are supported without changing the existing flat local layout.
3. No supplemental artifact is selected. The embedded tokenizer/template is
   sufficient for text; vision projectors and MTP sidecars remain separate.
4. Halo's pinned Lemonade 11.8.1 stable backend is llama.cpp build 10594 and
   does not contain `qwen4exp`. The standalone ROCm image is build 10711 at
   `9723942ad`, contains the architecture and required attention workaround,
   and is the initial baseline.
5. The next step is a monitored no-MTP 32K load and multi-message quality
   canary. Do not add vision, MTP, or longer context until that passes.
6. EngramHalo.cpp is the leading Strix-specific performance follow-up because
   it can keep the PLE table SSD-backed, but its moving branch needs a separate
   pinned image and qualification; it must not replace this baseline implicitly.
