# Third-Party Notices and Distribution Boundaries

## Audio8 source code

This repository derives from `Audio8-AI/Audio8_TTS` at commit
`3346560df718d33096ac2fef7e5c7984ee5248e6`.

The upstream repository declares the Apache License 2.0. Its `LICENSE` and
`NOTICE` files are retained. Files changed by Instavar are identified through
the Git history and this notice. Apache 2.0 requires recipients to receive the
licence, retain applicable notices and receive prominent notice of modified
files.

## Audio8 model weights

The model used for the bounded reproduction is
`Audio8/Audio8-TTS-Preview-0.6b` at revision
`1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17`. Its model card currently declares
Apache 2.0, and the upstream source repository states that its code and model
weights use Apache 2.0. The inspected model snapshot does not contain a separate
`LICENSE` file. Preserve the source repository's Apache licence and notices, and
recheck the model repository before redistributing weights or an adapter.

The weights are not included in this source repository. Users obtain them from
the model publisher. Pin the revision when reproducibility matters and review
the model card again before redistribution because repository metadata can
change after this review.

Loading this model with `trust_remote_code=True` executes Python files from the
pinned model snapshot. Review and pin those files as part of any packaged or
hosted release rather than treating the weight tensor as the only dependency.

## Training data and voices

This repository contains no National Speech Corpus recordings, transcripts,
prepared codec arrays or FEMALE_01 adapter. The ability to access a dataset is
not itself permission to redistribute it or impersonate a contributor.

Before training or publishing an adapter, verify separately:

- the dataset licence and redistribution terms;
- consent or another valid basis for adapting the represented voice;
- whether the corpus represents one person or several people;
- the intended commercial, research and geographic uses;
- obligations affecting generated speech and model distribution.

## Instavar modifications

Instavar added LoRA and adapter support to these upstream files:

- `audio8_tts_sft.py`
- `audio8_tts_infer.py`
- `requirements-train.txt`
- `README.md`

Instavar added these files:

- `audio8_tts_lora.sh`
- `requirements-lora-cuda.txt`
- `requirements-macos.txt`
- `THIRD_PARTY_NOTICES.md`
- repository tests and continuous-integration configuration

This document records a technical attribution review. It is not legal advice.
