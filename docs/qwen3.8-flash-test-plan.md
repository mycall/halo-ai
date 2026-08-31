Next is preparation before the large downloads:
1. Commit the new [Stage 5 plan](/home/michael/Projects/source/github/halo-ai/TODO.md), which is currently uncommitted.
2. Tomorrow, create a fresh system snapshot and confirm roughly 174 GiB of model storage plus working headroom is available.
3. Recheck and pin the fast-moving model and engine revisions.
4. Implement reusable, resumable acquisition and a separate Flash-Next runtime image—without altering qualified q38rocm build 213.
5. Download only the 87.06 GiB ROCmFP4 baseline first.
6. Perform a monitored, no-MTP 32K load and output-quality canary.
7. If it passes, acquire the size-matched 83.81 GiB Unsloth Q3 control and optional 2.28 GiB MTP model.
8. Run the interleaved ROCmFP4/Unsloth/DeepSeek comparison, emphasizing same-session speed, energy, memory, and TDP ratios.
9. Add vision and longer contexts only after the basic comparison is stable.
The very next implementation task is therefore the acquisition/catalog/runtime plumbing—not downloading the model yet.

