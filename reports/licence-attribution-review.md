# Licence and attribution review

Date: 2026-07-30

## Reviewed surfaces

This review covered three different things that should not be collapsed into a
single licence claim:

1. the [`Audio8-AI/Audio8_TTS`](https://github.com/Audio8-AI/Audio8_TTS)
   source repository at commit
   `3346560df718d33096ac2fef7e5c7984ee5248e6`;
2. the
   [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
   model snapshot at revision
   `1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`;
3. the recordings, transcripts, codec targets and voice represented by any
   user-supplied training dataset.

## Findings

The source repository includes an Apache License 2.0 file and an upstream
`NOTICE`. Its README states that its code and model weights use Apache 2.0. This
derivative retains the upstream Git history, `LICENSE` and `NOTICE`, identifies
the upstream commit, and adds prominent modification notices to the two changed
Python files.

The pinned Hugging Face model card declares `apache-2.0`. The inspected model
snapshot does not contain a separate `LICENSE` file. The model also includes
custom Python code loaded through `trust_remote_code=True`, so a packaged
release must pin and review that code as well as the tensors. The weights are
not included here and repository metadata should be checked again before any
redistribution.

No National Speech Corpus material, FEMALE_01 recording, prepared codec array
or trained adapter is included. A code or model licence does not grant rights
to a dataset or a person's voice. Those rights require a separate review for
the intended training, distribution, commercial use and geography.

## Distribution decision

Publishing this source-only derivative is technically consistent with the
reviewed Apache attribution requirements when the retained licence and notices
remain intact. Publishing model weights, a dataset or a voice adapter is a
separate decision and is outside this repository's distribution scope.

This is a technical attribution review, not legal advice.
