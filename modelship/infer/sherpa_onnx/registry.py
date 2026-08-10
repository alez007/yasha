"""Curated table of supported sherpa_onnx models, keyed by name. `model:` in
config must be a name here, or a local directory whose basename is one.

No network access and no `import sherpa_onnx` — must be importable without the
wheel installed, same as capabilities.py's LOADER_MODULES.
"""

from typing import NamedTuple


class RegistryFile(NamedTuple):
    """A file inside the extracted bundle, relative to its root. `sha256` is
    None for files too large to hash on every deploy."""

    path: str
    size: int
    sha256: str | None = None


class RegistryDir(NamedTuple):
    """A directory inside the extracted bundle, relative to its root."""

    path: str
    file_count: int


class SherpaOnnxRegistryEntry(NamedTuple):
    """`files`/`dirs` keys are sherpa's own `OfflineTtsKokoroModelConfig`
    attribute names. `lexicon` is the list form of sherpa's single comma-joined
    field. `voice_names[i]` is the OpenAI voice name for sid i."""

    tarball_url: str
    sha256: str
    family: str
    usecase: str
    files: dict[str, RegistryFile]
    dirs: dict[str, RegistryDir]
    lexicon: tuple[RegistryFile, ...]
    voice_names: tuple[str, ...]


_RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"

REGISTRY: dict[str, SherpaOnnxRegistryEntry] = {
    "kokoro-en-v0_19": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-en-v0_19.tar.bz2",
        sha256="912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
        family="kokoro",
        usecase="tts",
        files={
            "model": RegistryFile("model.onnx", 345_555_491),
            "tokens": RegistryFile(
                "tokens.txt", 1_078, "4f31c71282d14af4e926cd12462078fe9d20d00c589e63fe2750a8f56d6d7f7b"
            ),
            "voices": RegistryFile(
                "voices.bin", 5_755_904, "a372c67b056ef0b695c375d39b99630d23fb07ad4c8d87aa32a19a62fca523ad"
            ),
        },
        dirs={
            "data_dir": RegistryDir("espeak-ng-data", 355),
        },
        lexicon=(),
        voice_names=(
            "af",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ),
    ),
    # lexicon order/set matches upstream's own example (kokoro multi-lang needs
    # us-en + zh; gb-en is an alternative to us-en, not additive). dict/ and the
    # *-zh.fst rule files are unused optional extras, left out on purpose.
    "kokoro-multi-lang-v1_0": SherpaOnnxRegistryEntry(
        tarball_url=f"{_RELEASE_BASE}/kokoro-multi-lang-v1_0.tar.bz2",
        sha256="c133d26353d776da730870dac7da07dbfc9a5e3bc80cc5e8e83ab6e823be7046",
        family="kokoro",
        usecase="tts",
        files={
            "model": RegistryFile("model.onnx", 325_630_829),
            "tokens": RegistryFile(
                "tokens.txt", 687, "6ebb6bb288f20f3ae8d004d3c2ca27697da27c037d75e81a60e2a6a663f95425"
            ),
            "voices": RegistryFile(
                "voices.bin", 27_678_720, "8a77c0d397026208d22211f37670b5b3b11e03f190756b25a1d24041fced82a9"
            ),
        },
        dirs={
            "data_dir": RegistryDir("espeak-ng-data", 355),
        },
        lexicon=(
            RegistryFile("lexicon-us-en.txt", 5_956_885),
            RegistryFile("lexicon-zh.txt", 2_364_621),
        ),
        voice_names=(
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_heart",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
            "bf_alice",
            "bf_emma",
            "bf_isabella",
            "bf_lily",
            "bm_daniel",
            "bm_fable",
            "bm_george",
            "bm_lewis",
            "ef_dora",
            "em_alex",
            "ff_siwis",
            "hf_alpha",
            "hf_beta",
            "hm_omega",
            "hm_psi",
            "if_sara",
            "im_nicola",
            "jf_alpha",
            "jf_gongitsune",
            "jf_nezumi",
            "jf_tebukuro",
            "jm_kumo",
            "pf_dora",
            "pm_alex",
            "pm_santa",
            "zf_xiaobei",
            "zf_xiaoni",
            "zf_xiaoxiao",
            "zf_xiaoyi",
            "zm_yunjian",
            "zm_yunxi",
            "zm_yunxia",
            "zm_yunyang",
        ),
    ),
}


def registry_names() -> tuple[str, ...]:
    return tuple(REGISTRY)
