# Evidence archive

[Reference manual](../README.md)

These documents preserve the original conditions, measurements, and reasoning
behind earlier work. They are frozen records, not installation instructions or
a source of current defaults. Runtime pins, aliases, host state, and qualification
status can differ from the reference manual.

| Record | Scope |
| --- | --- |
| [Implementation notes](implementation-notes.md) | Original architecture plan, phased integration, and runtime experiments |
| [Host and runtime validation](validation-history.md) | Initial profile matrix, speech checks, and LongBench canary |
| [Halogen integration](halogen-flash-history.md) | Checkpoint integration, NPU preparation, and early serving trials |
| [Halogen evaluations](halogen-evaluations.md) | Runtime comparison, checkpoint memory, ingestion, and SciFact results |
| [Halogen upstream reviews](halogen-upstream-review.md) | Release research and local follow-up evidence |
| [Qwen evaluations](qwen-evaluations.md) | Runtime, draft, and quantization comparisons |
| [Flash-Next qualification record](flash-next-qualification.md) | GGUF/MTP preparation and Strix Vulkan trial matrix |
| [Planning notes](planning-notes.md) | Superseded research roadmap |

New trial artifacts belong in [results](../results/). Update the relevant manual
chapter with the resulting behavior or limitation instead of extending these logs.
