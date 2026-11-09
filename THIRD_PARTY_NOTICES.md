# Third-party components and dataset provenance

## Training recordings

- **Mozilla Common Voice 17.0, Japanese and English**: CC0-1.0 (public-domain dedication). Commercial use and reuse without attribution are allowed by that dedication. Original project terms: https://commonvoice.mozilla.org/en/terms and https://commonvoice.mozilla.org/dav/terms . CC0 explanation: https://creativecommons.org/publicdomain/zero/1.0/ . Versioned Parquet source: https://huggingface.co/datasets/fixie-ai/common_voice_17_0 . No raw corpus audio is included in the distributable package. Do not attempt to identify the speakers.
- `data/provenance.json` records the exact source revision, URLs and downloaded SHA-256 digests. `reports/split_manifest.json` records only hashed speaker/transcript IDs, file IDs and split/statistical metadata; not raw transcripts or speaker identities.

The no-attribution choice concerns the recording dataset. The following libraries and training tools retain their own licenses; this does not require an on-screen dataset credit in a stream or video.

## Additional event-recording lookup

Freesound recordings are considered only when the individual audio license is **CC0-1.0**. The mixed-license FSD50K catalog is a lookup reference, not a declaration that the entire dataset is CC0. Reference: Fonseca et al., *FSD50K: An Open Dataset of Human-Labeled Sound Events*, https://zenodo.org/records/4060432 (catalog/curation CC BY 4.0), mirrored at https://huggingface.co/datasets/Fhrozen/FSD50k . No FSD50K event annotations or standard splits are imported; the script independently selects clips from uploader titles/descriptions. `data/breath_cc0/manifest.json` stores each selected recording's CC0 license URL, original Freesound page, pinned audio mirror URL, SHA-256, selection label and uploader-disjoint partition. Raw recordings are excluded from Git and releases. Clip descriptions are weak event labels, not human-verified breath boundaries.

PyTorch (BSD-style license) is optional and used only to train temporal candidates; it is not imported by the inference application.

## Training-only automatic annotation models

- **prj-beatrice/japanese-hubert-base-phoneme-ctc-v4**, Apache-2.0: https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v4 . Based on rinna/japanese-hubert-base. ONNX Q4 conversion by air-air: https://huggingface.co/air-air/japanese-hubert-base-phoneme-ctc-v4-ONNX . Base teacher training data is described in that model's card and is not downloaded as this project's training recordings.
- **facebook/wav2vec2-lv-60-espeak-cv-ft**, Apache-2.0: https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft . ONNX conversion: https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX . Authors Qiantong Xu, Alexei Baevski, Michael Auli, *Simple and Effective Zero-shot Cross-lingual Phoneme Recognition* (2021), https://arxiv.org/abs/2109.11680 .

These teachers are used only for generating pseudo labels locally. They are not required by the runtime. Exact revisions are in the provenance file. Downloaded teacher weights stay in `cache/teachers` and are excluded from release archives.

## Runtime dependencies

Python (PSF), NumPy (BSD-3-Clause and bundled components), SciPy (BSD-3-Clause and bundled components), SoundFile (BSD-3-Clause, bundles libsndfile), python-sounddevice (MIT, bundles PortAudio), threadpoolctl (BSD-3-Clause), cffi (MIT), pycparser (BSD). The installed distributions include full license texts in their `*.dist-info` directories. Redistributed bundled executables should retain these texts. Training dependencies include scikit-learn (BSD-3-Clause), ONNX Runtime (MIT), Apache Arrow (Apache-2.0), Requests (Apache-2.0) and pytest (MIT).

## External specifications

- VRChat built-in animator parameters are read-only: https://creators.vrchat.com/avatars/animator-parameters/ . Kotoba uses only its own `Kotoba_*` parameters.
- VMC Protocol: https://protocol.vmc.info/english . Implemented endpoints: Blend/Val and Blend/Apply.

No VRChat SDK, Unity Editor binary, character art or third-party avatar is redistributed.
