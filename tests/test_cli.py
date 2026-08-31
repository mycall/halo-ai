from __future__ import annotations

import importlib.util
import json
import os
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib/halo_ai"))
SPEC = importlib.util.spec_from_file_location("halo_ai_cli", ROOT / "lib/halo_ai/cli.py")
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)

SPEECH_LANGUAGE_SPEC = importlib.util.spec_from_file_location(
    "halo_ai_speech_languages", ROOT / "lib/halo_ai/speech/speech_languages.py"
)
assert SPEECH_LANGUAGE_SPEC and SPEECH_LANGUAGE_SPEC.loader
speech_languages = importlib.util.module_from_spec(SPEECH_LANGUAGE_SPEC)
SPEECH_LANGUAGE_SPEC.loader.exec_module(speech_languages)


def make_config(root: Path) -> object:
    values = dict(cli.DEFAULTS)
    values.update(
        {
            "HALO_AI_STATE_DIR": str(root / "state"),
            "HALO_AI_CACHE_DIR": str(root / "cache"),
            "HALO_AI_MODELS_ROOT": str(root / "models"),
            "HALO_AI_CATALOG_DIR": str(ROOT / "config/models.d"),
            "HALO_AI_PRESET_DIR": str(ROOT / "config/request-presets.d"),
            "HALO_AI_INVENTORY_FILE": str(root / "state/inventory.json"),
            "DS4_KV_CACHE_DIR": str(root / "cache/ds4-kv"),
            "SPEECH_TEST_AUDIO_PATH": str(root / "cache/speech/input1.wav"),
        }
    )
    return cli.Config(values, ())


def gguf_fixture(payload: bytes = b"") -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + payload


class ConfigurationTests(unittest.TestCase):
    def test_current_runtime_defaults_select_lemonade_11_8_and_rocm_10(self) -> None:
        self.assertEqual(cli.LEMONADE_VERSION, "11.8.1")
        self.assertEqual(
            cli.DEFAULTS["LEMONADE_IMAGE"],
            "ghcr.io/lemonade-sdk/lemonade-server@"
            "sha256:824359e8633d3cde4afb2c32609930758f4e71424d71ad58f26432a8bb1092cb",
        )
        self.assertEqual(
            cli.DEFAULTS["LLAMACPP_IMAGE"],
            "docker.io/kyuz0/amd-strix-halo-toolboxes@"
            "sha256:65fcb5855f6186b8a6ddf56e89abf743fa0fa91ce76394ad2bd7ed7bc9cd10b6",
        )
        example = cli.parse_env_file(ROOT / "config/halo-ai.env.example")
        self.assertEqual(example["LEMONADE_IMAGE"], cli.DEFAULTS["LEMONADE_IMAGE"])
        self.assertEqual(example["LLAMACPP_IMAGE"], cli.DEFAULTS["LLAMACPP_IMAGE"])

    def test_opencode_exposes_native_context_qwen3_8_vision_alias(self) -> None:
        document = json.loads((ROOT / "config/opencode.json").read_text(encoding="utf-8"))
        self.assertEqual(document["enabled_providers"], ["halo-ai"])
        self.assertEqual(set(document["provider"]), {"halo-ai"})
        provider = document["provider"]["halo-ai"]
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:8000/v1")
        self.assertEqual(
            set(provider["models"]), {"ds4", "qwen3.8df2", "qwen3.8fp4", "qwen3.8fp8"},
        )
        for model_id in ("qwen3.8df2", "qwen3.8fp4", "qwen3.8fp8"):
            variants = provider["models"][model_id]["variants"]
            self.assertEqual(set(variants), {"none", "low", "medium", "xhigh"})
            self.assertEqual(variants["medium"]["reasoningEffort"], "medium")
            self.assertNotIn("high", variants)
        model = provider["models"]["qwen3.8df2"]
        self.assertEqual(model["limit"]["context"], 262_144)
        self.assertTrue(model["attachment"])
        self.assertEqual(model["modalities"]["input"], ["text", "image"])

    def test_start_help_describes_switch(self) -> None:
        parser = cli.build_parser()
        commands = next(
            action for action in parser._actions
            if isinstance(action, __import__("argparse")._SubParsersAction)
        )
        help_text = commands.choices["start"].format_help()
        self.assertIn("--switch", help_text)
        self.assertIn("stop a conflicting managed inference runtime", help_text)

    def test_configuration_is_data_not_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "must-not-exist"
            config = Path(temporary) / "config.env"
            config.write_text(f"HALO_AI_RUN_USER=$(touch {marker})\n", encoding="utf-8")
            parsed = cli.parse_env_file(config)
            self.assertEqual(parsed["HALO_AI_RUN_USER"], f"$(touch {marker})")
            self.assertFalse(marker.exists())

    def test_explicit_config_resolves_workspace_lookup_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "config/models.d"
            presets = root / "config/request-presets.d"
            catalog.mkdir(parents=True)
            presets.mkdir(parents=True)
            workspace = root / "workspace.env"
            workspace.write_text(
                "HALO_AI_RUN_USER=\n"
                "HALO_AI_CATALOG_DIR=config/models.d\n"
                "HALO_AI_PRESET_DIR=config/request-presets.d\n",
                encoding="utf-8",
            )
            loaded = cli.load_config(str(workspace))
            self.assertEqual(loaded.path("HALO_AI_CATALOG_DIR"), catalog)
            self.assertEqual(loaded.path("HALO_AI_PRESET_DIR"), presets)

    def test_empty_run_user_selects_only_a_non_root_caller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            config.values["HALO_AI_RUN_USER"] = ""
            with mock.patch.object(cli.os, "geteuid", return_value=1000):
                cli.assert_operator(config)
            with mock.patch.object(cli.os, "geteuid", return_value=0):
                with self.assertRaisesRegex(cli.HaloError, "non-root operator"):
                    cli.assert_operator(config)

    def test_gtt_candidates_must_be_strictly_increasing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            config.values["HALO_AI_GTT_CANDIDATES_GIB"] = "112,112,118"
            with self.assertRaises(cli.HaloError):
                cli.validate_config(config)

    def test_llamacpp_rocm_bin_is_constrained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            for value in ("builtin", "latest", "b10334"):
                config.values["LEMONADE_LLAMACPP_ROCM_BIN"] = value
                cli.validate_config(config)
            config.values["LEMONADE_LLAMACPP_ROCM_BIN"] = "main"
            with self.assertRaises(cli.HaloError):
                cli.validate_config(config)

    def test_lemonade_rocm_environment_is_allowlisted_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            config.values["LEMONADE_ROCM_CHANNEL"] = "nightly"
            config.values["LEMONADE_GPU_MAX_HW_QUEUES"] = "1"
            cli.validate_config(config)
            self.assertEqual(
                cli.lemonade_runtime_environment(config), {"GPU_MAX_HW_QUEUES": "1"},
            )
            config.values["LEMONADE_GPU_MAX_HW_QUEUES"] = "0"
            with self.assertRaises(cli.HaloError):
                cli.validate_config(config)

    def test_deepseek_think_max_preset_uses_official_sampling_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            preset = cli.load_presets(config)["deepseek-v4-think-max"]
        self.assertEqual(preset["request"]["reasoning_effort"], "max")
        self.assertEqual(preset["request"]["temperature"], 1.0)
        self.assertEqual(preset["request"]["top_p"], 1.0)
        self.assertIn("context.393216", preset["requires"])


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.config = make_config(Path(temporary))
            self.catalog = cli.load_catalog(self.config)

    def test_checked_in_catalog_validates(self) -> None:
        self.assertEqual(len(self.catalog.models), 11)
        self.assertEqual(len(self.catalog.profiles), 37)
        self.assertEqual(
            self.catalog.profile_aliases,
            {
                "ds4": "ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max",
                "qwen3.8df2": "qwen3.8-27b-q6xl-strix-vision-dflash2",
                "qwen3.8fp4": "qwen3.8-27b-rocmfp4-baseline",
                "qwen3.8fp8": "qwen3.8-27b-rocmfp8-baseline",
            },
        )

    def test_ds4_alias_resolves_to_canonical_think_max_profile(self) -> None:
        canonical = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max"
        ]
        self.assertIs(cli.resolve_profile(self.catalog, "ds4"), canonical)
        self.assertIs(cli.resolve_profile(self.catalog, canonical["id"]), canonical)

    def test_qwen3_8_aliases_resolve_to_baseline_profiles(self) -> None:
        for alias, canonical_id in (
            ("qwen3.8fp4", "qwen3.8-27b-rocmfp4-baseline"),
            ("qwen3.8fp8", "qwen3.8-27b-rocmfp8-baseline"),
        ):
            with self.subTest(alias=alias):
                canonical = self.catalog.profiles[canonical_id]
                self.assertIs(cli.resolve_profile(self.catalog, alias), canonical)

    def test_qwen3_8df2_alias_resolves_to_q6_vision_dflash2(self) -> None:
        canonical = self.catalog.profiles["qwen3.8-27b-q6xl-strix-vision-dflash2"]
        self.assertIs(cli.resolve_profile(self.catalog, "qwen3.8df2"), canonical)
        self.assertEqual(canonical["model"], "qwen3.8-27b-ud-q6-k-xl")
        self.assertEqual(canonical["features"], ["vision", "dflash"])
        self.assertEqual(canonical["draft_model"], "qwen3.8-27b-dflash2-q4-k-m")

    def test_showing_ds4_alias_preserves_alias_and_canonical_identity(self) -> None:
        arguments = __import__("argparse").Namespace(
            profiles_action="show", profile_id="ds4",
        )
        output = __import__("io").StringIO()
        with (
            mock.patch.object(cli, "profile_availability", return_value=(True, "ready")),
            __import__("contextlib").redirect_stdout(output),
        ):
            self.assertEqual(
                cli.command_profiles(self.config, self.catalog, arguments), 0,
            )
        document = json.loads(output.getvalue())
        self.assertEqual(document["id"], "ds4")
        self.assertEqual(
            document["alias_for"],
            "ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max",
        )
        self.assertEqual(document["context"], 393_216)

    def test_profile_alias_must_target_a_canonical_profile(self) -> None:
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog(
                self.catalog.models, self.catalog.profiles,
                {"broken": "missing-profile"},
            )

    def test_profile_alias_cannot_shadow_a_canonical_profile(self) -> None:
        canonical = "ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max"
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog(
                self.catalog.models, self.catalog.profiles,
                {canonical: canonical},
            )

    def test_mtp_cannot_combine_with_vision(self) -> None:
        models = {key: dict(value) for key, value in self.catalog.models.items()}
        profile = dict(self.catalog.profiles["qwen3.6-27b-q8xl-mtp-llamacpp"])
        profile["features"] = ["mtp", "vision"]
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog(models, {profile["id"]: profile})

    def test_hash_length_is_enforced(self) -> None:
        model = json.loads(json.dumps(self.catalog.models["qwen3.6-35b-a3b-q8xl"]))
        model["files"][0]["sha256"] = "bad"
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog({model["id"]: model}, {})

    def test_catalog_matches_documented_expected_manifest(self) -> None:
        documented = {}
        for digest, size, relative in __import__("re").findall(
            r"^([0-9a-f]{64}) ([0-9]+) ((?:antirez|unsloth|facebook|julianmb|incoai)/[^\n]+\.(?:gguf|jinja|json|safetensors|model))$",
            (ROOT / "docs/halo-ai.md").read_text(encoding="utf-8"),
            __import__("re").MULTILINE,
        ):
            documented[relative] = (int(size), digest)
        cataloged = {
            item["path"]: (item["bytes"], item["sha256"])
            for model in self.catalog.models.values()
            for item in model["files"]
        }
        self.assertEqual(cataloged, documented)

    def test_qwen_profile_mounts_and_selects_pinned_template(self) -> None:
        profile = self.catalog.profiles["qwen3.6-27b-q8xl-lemonade"]
        model = self.catalog.models[profile["model"]]
        self.assertEqual(profile["chat_template"], "nonthinking")
        self.assertEqual(
            cli.container_template_path(model, profile),
            "/models/templates/qwen3.6-nonthinking.jinja",
        )
        command = cli.render_container(self.config, self.catalog, profile)
        rendered = __import__("shlex").join(command)
        self.assertIn("local.halo-ai.runtime-spec=", rendered)
        self.assertIn("dst=/models/templates/qwen3.6-nonthinking.jinja,ro", rendered)
        self.assertIn("dst=/models/templates/qwen3.6-thinking.jinja,ro", rendered)
        self.assertIn(
            "dst=/models/extra/qwen3.6-27b-q8xl/Qwen3.6-27B-UD-Q8_K_XL.gguf,ro",
            rendered,
        )
        self.assertIn(
            "dst=/models/extra/qwen3.6-27b-q8xl-vision/Qwen3.6-27B-UD-Q8_K_XL.gguf,ro",
            rendered,
        )
        self.assertIn(
            "dst=/models/extra/qwen3.6-27b-q8xl-vision/mmproj-F32.gguf,ro",
            rendered,
        )

    def test_lemonade_11_8_mounts_distinct_cache_and_config_volumes(self) -> None:
        profile = self.catalog.profiles["qwen3.6-27b-q8xl-lemonade"]
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        self.assertIn(
            "halo-lemonade-config:/opt/lemonade/.cache/lemonade:U", rendered,
        )
        self.assertIn(
            "halo-lemonade-state:/opt/lemonade/.config/lemonade:U", rendered,
        )

    def test_lemonade_11_8_upgrade_backs_up_legacy_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            legacy = root / "legacy-volume"
            legacy.mkdir()
            (legacy / "config.json").write_text('{"port": 13305}\n', encoding="utf-8")
            (legacy / "user_models.json").write_text('{"models": []}\n', encoding="utf-8")
            (legacy / "unrelated.bin").write_bytes(b"not configuration")
            inspected = mock.MagicMock(returncode=0, stdout=str(legacy) + "\n")
            with mock.patch.object(cli, "podman", return_value=inspected):
                backup = cli.backup_lemonade_legacy_config(config)
            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertEqual(
                (backup / "config.json").read_text(encoding="utf-8"),
                '{"port": 13305}\n',
            )
            self.assertEqual(
                (backup / "user_models.json").read_text(encoding="utf-8"),
                '{"models": []}\n',
            )
            self.assertFalse((backup / "unrelated.bin").exists())
            manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["lemonade_upgrade"], "11.8.1")
            self.assertEqual(set(manifest["files"]), {"config.json", "user_models.json"})

    def test_lemonade_storage_preparation_creates_every_managed_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))

            def fake_podman(arguments, **_kwargs):
                return mock.MagicMock(returncode=1 if arguments[1] == "exists" else 0)

            with (
                mock.patch.object(cli, "backup_lemonade_legacy_config") as backup,
                mock.patch.object(cli, "podman", side_effect=fake_podman) as podman,
            ):
                cli.prepare_lemonade_storage(config)
            backup.assert_called_once_with(config)
            created = [
                call.args[0][-1] for call in podman.call_args_list
                if call.args[0][:2] == ["volume", "create"]
            ]
            self.assertEqual(
                created,
                [name for name, _component, _destination in cli.LEMONADE_VOLUME_MOUNTS],
            )

    def test_lemonade_runtime_spec_is_shared_across_profiles(self) -> None:
        first = self.catalog.profiles["qwen3.6-27b-q8xl-lemonade"]
        second = self.catalog.profiles["qwen3.6-35b-a3b-q8xl-lemonade"]
        labels = []
        for profile in (first, second):
            command = cli.render_container(self.config, self.catalog, profile)
            labels.append(next(item for item in command if item.startswith("local.halo-ai.runtime-spec=")))
        self.assertEqual(labels[0], labels[1])

    def test_lemonade_runtime_spec_tracks_backend_policy(self) -> None:
        builtin = cli.lemonade_runtime_spec(self.config, self.catalog)
        self.config.values["LEMONADE_LLAMACPP_ROCM_BIN"] = "b10334"
        pinned = cli.lemonade_runtime_spec(self.config, self.catalog)
        self.assertNotEqual(builtin, pinned)

    def test_ds4_render_forces_rocm_and_container_listener(self) -> None:
        profile = self.catalog.profiles["ds4-deepseek-v4-flash-hybrid"]
        command = cli.render_container(self.config, self.catalog, profile)
        rendered = __import__("shlex").join(command)
        self.assertIn("ds4-server", command)
        self.assertIn("--rocm --host 0.0.0.0 --port 8000", rendered)
        self.assertIn("-p 127.0.0.1:8000:8000", rendered)
        self.assertIn("--group-add keep-groups", rendered)
        self.assertNotIn("--group-add video", rendered)

    def test_ds4_kv_profile_mounts_private_cache_and_explicit_policy(self) -> None:
        profile = self.catalog.profiles["ds4-deepseek-v4-flash-hybrid-kv"]
        rendered = __import__("shlex").join(cli.render_container(self.config, self.catalog, profile))
        self.assertIn("dst=/var/cache/ds4-kv", rendered)
        self.assertIn("--kv-disk-dir /var/cache/ds4-kv", rendered)
        self.assertIn("--kv-disk-space-mb 8192", rendered)
        self.assertIn("--kv-cache-min-tokens 512", rendered)
        self.assertIn("--kv-cache-reject-different-quant", rendered)

    def test_speech_profile_uses_direct_facebook_model_path(self) -> None:
        profile = self.catalog.profiles["seamless-m4t-v2-large-speech"]
        command = cli.render_container(self.config, self.catalog, profile)
        rendered = __import__("shlex").join(command)
        expected = self.config.path("HALO_AI_MODELS_ROOT") / "facebook/seamless-m4t-v2-large"
        self.assertIn(f"src={expected},dst=/models/seamless-m4t-v2-large,ro", rendered)
        self.assertNotIn("/huggingface/facebook/", rendered)
        self.assertIn("-p 127.0.0.1:7860:7860", rendered)
        self.assertIn("--group-add keep-groups", rendered)
        self.assertIn("SPEECH_MODEL_REVISION=5f8cc790b19fc3f67a61c105133b20b34e3dcb76", rendered)

    def test_speech_language_set_exposes_every_bidirectional_speech_code(self) -> None:
        expected = set(speech_languages.LANGUAGE_NAMES_BY_CODE)
        generation_config = type("GenerationConfig", (), {
            "text_decoder_lang_to_code_id": dict.fromkeys(expected | {"text_only"}, 1),
            "t2u_lang_code_to_id": dict.fromkeys(expected | {"no_vocoder"}, 1),
            "vocoder_lang_code_to_id": dict.fromkeys(expected | {"no_t2u"}, 1),
        })()
        exposed = speech_languages.speech_input_output_languages(generation_config)
        self.assertEqual(set(exposed.values()), expected)
        self.assertEqual(exposed["Ukrainian"], "ukr")
        self.assertEqual(len(exposed), 36)

    def test_dspark_render_selects_current_speculation_mode_and_no_mmap(self) -> None:
        profile = self.catalog.profiles["deepseek-v4-flash-0731-iq3xxs-dspark-llamacpp"]
        command = cli.render_container(self.config, self.catalog, profile)
        rendered = __import__("shlex").join(command)
        self.assertIn("--spec-type draft-dspark", rendered)
        self.assertIn("--model-draft /models/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", rendered)
        self.assertIn("--load-mode none", rendered)

    def test_rocmfpx_baseline_is_distinct_unassisted_vulkan_engine(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-rocmfp4-baseline"]
        model = self.catalog.models[profile["model"]]
        command = cli.render_container(self.config, self.catalog, profile)
        rendered = __import__("shlex").join(command)
        self.assertEqual(model["modalities"], ["text"])
        self.assertEqual(model["engines"], ["rocmfpx"])
        self.assertIn("--name halo-rocmfpx", rendered)
        self.assertIn("--device=/dev/dri", command)
        self.assertNotIn("--device=/dev/kfd", command)
        self.assertIn("--device Vulkan0", rendered)
        self.assertIn("--cache-type-k q8_0 --cache-type-v turbo4", rendered)
        self.assertIn("--seed 1 --temp 0", rendered)
        self.assertIn("--spec-type none", rendered)
        self.assertNotIn("draft-mtp", rendered)
        self.assertNotIn("--mmproj", command)
        self.assertIn("localhost/halo-ai-rocmfpx:v1.0.0", command)
        self.assertIn("ROCmFP4-FAST.gguf,ro", rendered)

    def test_q6_strix_profiles_isolate_baseline_dflash_and_vision(self) -> None:
        baseline = __import__("shlex").join(cli.render_container(
            self.config, self.catalog,
            self.catalog.profiles["qwen3.8-27b-q6xl-strix-baseline"],
        ))
        dflash = __import__("shlex").join(cli.render_container(
            self.config, self.catalog,
            self.catalog.profiles["qwen3.8-27b-q6xl-strix-dflash2"],
        ))
        vision = __import__("shlex").join(cli.render_container(
            self.config, self.catalog,
            self.catalog.profiles["qwen3.8-27b-q6xl-strix-vision"],
        ))
        vision_dflash = __import__("shlex").join(cli.render_container(
            self.config, self.catalog,
            self.catalog.profiles["qwen3.8-27b-q6xl-strix-vision-dflash2"],
        ))
        for rendered in (baseline, dflash, vision, vision_dflash):
            self.assertIn("--name halo-strixvulkan", rendered)
            self.assertIn("--device=/dev/dri", rendered)
            self.assertNotIn("--device=/dev/kfd", rendered)
            self.assertIn("--device Vulkan0 --gpu-layers all --fit off", rendered)
            self.assertIn("--threads 16 --threads-batch 32", rendered)
            self.assertIn("--cache-type-k f16 --cache-type-v f16", rendered)
        self.assertIn("--spec-type none", baseline)
        self.assertNotIn("draft.gguf", baseline)
        self.assertIn("dst=/models/draft.gguf,ro", dflash)
        self.assertIn("--spec-type draft-dflash", dflash)
        self.assertIn("--spec-draft-model /models/draft.gguf", dflash)
        self.assertIn("--spec-draft-n-max 7 --spec-draft-ngl all", dflash)
        self.assertIn("--mmproj /models/mmproj-BF16.gguf", vision)
        self.assertIn("--spec-type none", vision)
        self.assertNotIn("draft-dflash", vision)
        self.assertIn("--mmproj /models/mmproj-BF16.gguf", vision_dflash)
        self.assertIn("--spec-type draft-dflash", vision_dflash)
        self.assertIn("dst=/models/draft.gguf,ro", vision_dflash)

    def test_q6_lemonade_profile_reuses_q6_and_bf16_vision_artifacts(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-q6xl-vision-lemonade"]
        model = self.catalog.models[profile["model"]]
        self.assertEqual(model["engines"], ["lemonade", "strixvulkan"])
        self.assertEqual(profile["context"], 262_144)
        self.assertEqual(profile["features"], ["vision"])
        self.assertFalse(profile["thinking"])
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        self.assertIn("--name halo-lemonade", rendered)
        self.assertIn("--device=/dev/kfd", rendered)
        self.assertIn("--device=/dev/dri", rendered)
        self.assertIn("--group-add keep-groups", rendered)
        self.assertIn(
            "dst=/models/extra/qwen3.8-27b-ud-q6-k-xl-vision/"
            "Qwen3.8-27B-UD-Q6_K_XL.gguf,ro",
            rendered,
        )
        self.assertIn(
            "dst=/models/extra/qwen3.8-27b-ud-q6-k-xl-vision/mmproj-BF16.gguf,ro",
            rendered,
        )
        payload = cli.lemonade_load_payload(profile)
        self.assertEqual(payload["ctx_size"], 262_144)
        self.assertEqual(payload["llamacpp_backend"], "rocm")
        self.assertIn("--batch-size 4096 --ubatch-size 4096", payload["llamacpp_args"])
        self.assertIn("--cache-type-k f16 --cache-type-v f16", payload["llamacpp_args"])
        self.assertIn("--spec-type none", payload["llamacpp_args"])
        with mock.patch.object(cli, "image_identity", return_value=cli.LEMONADE_IMAGE_DIGEST):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(plan["selected_artifact_classes"], ["model", "vision", "runtime"])
        self.assertNotIn("vision", plan["excluded_artifact_classes"])

    def test_q6_lemonade_dflash_profile_registers_existing_local_artifacts(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-q6xl-vision-dflash2-lemonade"]
        available, reason = cli.profile_availability(self.config, self.catalog, profile)
        self.assertFalse(available)
        self.assertIn("expected 81, got 58", reason)
        self.assertIn("PR #27342", reason)
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        root = "/models/registered/qwen3.8-27b-q6xl-vision-dflash2-lemonade"
        self.assertIn(f"dst={root}/main,ro", rendered)
        self.assertIn(f"dst={root}/mmproj-BF16.gguf,ro", rendered)
        self.assertIn(f"dst={root}/dflash-Qwen3.8-27B-DFlash2-Q4_K_M.gguf,ro", rendered)
        self.assertNotIn("/models/extra/qwen3.8-27b-dflash2", rendered)
        registration = cli.lemonade_dflash_registration(self.config, self.catalog, profile)
        self.assertEqual(registration["source"], "local_path")
        self.assertEqual(registration["model_name"], f"user.{profile['id']}")
        self.assertEqual(set(registration["checkpoints"]), {"main", "mmproj", "draft"})
        self.assertIn("dflash", registration["labels"])
        payload = cli.lemonade_load_payload(profile)
        self.assertIn("--spec-type draft-dflash", payload["llamacpp_args"])
        self.assertIn("--spec-draft-n-max 5", payload["llamacpp_args"])
        self.assertNotIn("--model-draft", payload["llamacpp_args"])
        with mock.patch.object(cli, "image_identity", return_value=cli.LEMONADE_IMAGE_DIGEST):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(
            plan["selected_artifact_classes"], ["model", "vision", "dflash2", "runtime"],
        )
        self.assertEqual(len(plan["draft_model"]["files"]), 1)

    def test_lemonade_native_ds4_candidate_records_failed_streaming_gate(self) -> None:
        profile = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-lemonade-native"
        ]
        model = self.catalog.models[profile["model"]]
        self.assertEqual(model["engines"], ["ds4", "lemonade"])
        self.assertEqual(profile["features"], ["native-ds4"])
        available, reason = cli.profile_availability(self.config, self.catalog, profile)
        self.assertFalse(available)
        self.assertIn("forces --ssd-streaming", reason)
        self.assertIn("selected expert id -1", reason)

        registration = cli.lemonade_native_ds4_registration(
            self.config, self.catalog, profile,
        )
        self.assertEqual(registration["recipe"], "ds4")
        self.assertEqual(registration["source"], "local_path")
        self.assertEqual(registration["model_name"], f"user.{profile['id']}")
        self.assertEqual(set(registration["checkpoints"]), {"main"})
        self.assertTrue(registration["checkpoints"]["main"].endswith(".gguf"))

        payload = cli.lemonade_load_payload(profile, registration["model_name"])
        self.assertEqual(payload["ctx_size"], 32_768)
        self.assertEqual(payload["ds4_args"], "--prefill-chunk 2048")
        self.assertNotIn("llamacpp_backend", payload)
        self.assertNotIn("--ssd-streaming", payload["ds4_args"])

        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        destination = registration["checkpoints"]["main"]
        self.assertIn(f"dst={destination},ro", rendered)
        self.assertNotIn(
            "/models/extra/deepseek-v4-flash-ds4-hybrid", rendered,
        )

    def test_q6_dflash_acquisition_includes_only_exact_companion(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-q6xl-strix-dflash2"]
        with mock.patch.object(cli, "strixvulkan_image_valid", return_value=True):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(
            plan["selected_artifact_classes"],
            ["q6-xl", "dflash2", "strixvulkan-runtime"],
        )
        self.assertEqual(len(plan["model"]["files"]), 1)
        self.assertEqual(len(plan["draft_model"]["files"]), 1)
        self.assertTrue(plan["draft_model"]["files"][0]["destination"].endswith(
            "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
        ))
        self.assertEqual(plan["runtime"]["source_commit"], cli.STRIXVULKAN_SOURCE_COMMIT)
        self.assertEqual(plan["runtime"]["image_digest"], cli.STRIXVULKAN_IMAGE_DIGEST)

    def test_strixvulkan_image_requires_digest_and_source_labels(self) -> None:
        labels = {
            "org.opencontainers.image.source": "https://github.com/Nathanw1014/strix-halo-llamacpp",
            "org.opencontainers.image.title": "strix-halo-llamacpp-vulkan",
        }
        with (
            mock.patch.object(cli, "image_identity", return_value=cli.STRIXVULKAN_IMAGE_DIGEST),
            mock.patch.object(cli, "image_labels", return_value=labels),
        ):
            self.assertTrue(cli.strixvulkan_image_valid("fixture"))
        with (
            mock.patch.object(cli, "image_identity", return_value="sha256:" + "0" * 64),
            mock.patch.object(cli, "image_labels", return_value=labels),
        ):
            self.assertFalse(cli.strixvulkan_image_valid("fixture"))

    def test_vision_dflash_profile_records_nmax5_qualification_policy(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-q6xl-strix-vision-dflash2"]
        self.assertEqual(profile["risk"], "experimental")
        self.assertIn("nmax5-equal-history", profile["proposal_policy"])
        self.assertEqual(profile["settings"]["spec_draft_n_max"], 5)
        with mock.patch.object(cli, "strixvulkan_image_valid", return_value=True):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(
            plan["selected_artifact_classes"],
            ["q6-xl", "vision", "dflash2", "strixvulkan-runtime"],
        )

    def test_vision_dflash_diagnostic_profiles_select_exact_drafters(self) -> None:
        expected = {
            "qwen3.8-27b-q6xl-strix-vision-dflash2-bf16":
                "Qwen3.8-27B-DFlash2-BF16.gguf",
            "qwen3.8-27b-q6xl-strix-vision-dflash2-q8":
                "Qwen3.8-27B-DFlash2-Q8_0.gguf",
        }
        for profile_id, filename in expected.items():
            with self.subTest(profile=profile_id):
                profile = self.catalog.profiles[profile_id]
                rendered = __import__("shlex").join(
                    cli.render_container(self.config, self.catalog, profile)
                )
                self.assertIn("--mmproj /models/mmproj-BF16.gguf", rendered)
                self.assertIn("--spec-type draft-dflash", rendered)
                self.assertIn("dst=/models/draft.gguf,ro", rendered)
                draft = self.catalog.models[profile["draft_model"]]
                self.assertTrue(draft["files"][0]["path"].endswith(filename))

    def test_rocmfpx_profile_acquisition_excludes_every_optional_class(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-rocmfp4-baseline"]
        with mock.patch.object(cli, "rocmfpx_image_valid", return_value=True):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(plan["selected_artifact_classes"], ["fp4", "rocmfpx-runtime"])
        self.assertEqual(
            plan["excluded_artifact_classes"],
            ["fp8", "npu", "vision", "bf16", "reference"],
        )
        self.assertEqual(len(plan["model"]["files"]), 1)
        selected = plan["model"]["files"][0]
        self.assertTrue(selected["destination"].endswith("Qwen3.8-27B-ROCmFP4-FAST.gguf"))
        self.assertEqual(selected["bytes"], 14_562_236_384)
        serialized = json.dumps(plan).lower()
        for forbidden in ("rocmfp8.gguf", "q4nx", "mmproj", "safetensors", "reference.gguf"):
            self.assertNotIn(forbidden, serialized)

    def test_rocmfpx_mtp_uses_strict_qwen_verification_and_no_new_model(self) -> None:
        profile = self.catalog.profiles["qwen3.8-27b-rocmfp4-mtp"]
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        self.assertIn("--spec-type draft-mtp", rendered)
        self.assertIn("--spec-mtp-strict-qwen", rendered)
        self.assertIn("--spec-draft-n-max 6", rendered)
        self.assertIn("--spec-draft-p-min 0.6", rendered)
        self.assertNotIn("--spec-type none", rendered)
        with mock.patch.object(cli, "rocmfpx_image_valid", return_value=True):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
        self.assertEqual(len(plan["model"]["files"]), 1)
        self.assertEqual(plan["model"]["additional_download_bytes"], 14_562_236_384)
        self.assertIn("in-gguf-mtp", plan["selected_artifact_classes"])
        self.assertNotIn("q4nx", json.dumps(plan).lower())

    def test_rocmfpx_q5_draft_candidates_isolate_cache_and_policy(self) -> None:
        compressed = self.catalog.profiles["qwen3.8-27b-rocmfp4-mtp-q5-draft"]
        conservative = self.catalog.profiles[
            "qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft"
        ]
        compressed_rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, compressed)
        )
        conservative_rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, conservative)
        )
        for rendered in (compressed_rendered, conservative_rendered):
            self.assertIn("--spec-draft-type-k q5_1", rendered)
            self.assertIn("--spec-draft-type-v q5_1", rendered)
            self.assertIn("--spec-mtp-strict-qwen", rendered)
        self.assertIn("--spec-draft-n-max 6 --spec-draft-p-min 0.6", compressed_rendered)
        self.assertIn("--spec-draft-n-max 2 --spec-draft-p-min 0.85", conservative_rendered)

    def test_rocmfpx_mtp_profile_gates_reflect_quality_policy(self) -> None:
        expected_matches = {
            "qwen3.8-27b-rocmfp4-mtp": "7 of 13",
            "qwen3.8-27b-rocmfp4-mtp-q5-draft": "7 of 13",
            "qwen3.8-27b-rocmfp8-mtp": "4 of 13",
        }
        for profile_id, expected in expected_matches.items():
            ready, reason = cli.profile_availability(
                self.config, self.catalog, self.catalog.profiles[profile_id],
            )
            self.assertFalse(ready)
            self.assertIn(expected, reason)
        conservative = self.catalog.profiles[
            "qwen3.8-27b-rocmfp4-mtp-conservative-q5-draft"
        ]
        self.assertEqual(conservative["risk"], "experimental")
        self.assertNotIn("gate", conservative)
        self.assertEqual(
            self.catalog.profile_aliases["qwen3.8fp4"],
            "qwen3.8-27b-rocmfp4-baseline",
        )

    def test_ds4_dspark_profile_selects_only_exact_antirez_support(self) -> None:
        control = self.catalog.profiles["ds4-deepseek-v4-flash-hybrid"]
        candidate = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-16k"
        ]
        control_rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, control)
        )
        candidate_rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, candidate)
        )
        self.assertNotIn("DSpark-support", control_rendered)
        self.assertNotIn("--dspark", control_rendered)
        self.assertIn("DeepSeek-V4-Flash-DSpark-support-0731.gguf", candidate_rendered)
        self.assertIn("-e DS4_DSPARK_STATS=1", candidate_rendered)
        self.assertIn("--dspark --dspark-confidence 0.7", candidate_rendered)
        self.assertIn("--ctx 16384 --prefill-chunk 1024", candidate_rendered)
        with mock.patch.object(cli, "ds4_image_valid", return_value=True):
            control_plan = cli.profile_acquisition_plan(self.config, self.catalog, control)
            candidate_plan = cli.profile_acquisition_plan(self.config, self.catalog, candidate)
        self.assertEqual([item["role"] for item in control_plan["model"]["files"]], ["main"])
        self.assertEqual(
            [item["role"] for item in candidate_plan["model"]["files"]],
            ["main", "dspark"],
        )
        self.assertNotIn("unsloth", json.dumps(candidate_plan).lower())
        self.assertEqual(candidate_plan["runtime"]["release"], "b0001")
        self.assertEqual(candidate_plan["runtime"]["ds4_commit"], cli.DS4_SOURCE_COMMIT)
        self.assertEqual(candidate_plan["runtime"]["engine_archive_sha256"], cli.DS4_RELEASE_SHA256)

    def test_optional_dspark_does_not_disable_ds4_control(self) -> None:
        model = self.catalog.models["deepseek-v4-flash-ds4-hybrid"]
        control = self.catalog.profiles["ds4-deepseek-v4-flash-hybrid"]
        candidate = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-16k"
        ]
        with mock.patch.object(cli, "verify_model") as verify:
            verify.return_value = {"valid": True}
            with mock.patch.object(cli, "ds4_image_valid", return_value=True):
                ready, _reason = cli.profile_availability(
                    self.config, self.catalog, control,
                )
                self.assertTrue(ready)
                self.assertEqual(verify.call_args.args[3], {"main"})
                candidate_ready, reason = cli.profile_availability(
                    self.config, self.catalog, candidate,
                )
                self.assertTrue(candidate_ready, reason)
                self.assertEqual(verify.call_args.args[3], {"main", "dspark"})
        self.assertEqual(
            {item["role"] for item in model["files"]}, {"main", "dspark"},
        )

    def test_ds4_dspark_128k_is_separate_from_qualified_16k_rollback(self) -> None:
        rollback = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-16k"
        ]
        candidate = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-128k"
        ]
        self.assertEqual(rollback["context"], 16_384)
        self.assertEqual(candidate["context"], 131_072)
        self.assertEqual(candidate["risk"], "experimental")
        self.assertEqual(candidate["features"], ["dspark", "kv-cache"])
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, candidate)
        )
        self.assertIn("--ctx 131072 --prefill-chunk 1024", rendered)
        self.assertIn("--dspark --dspark-confidence 0.7", rendered)
        self.assertIn("--kv-disk-dir /var/cache/ds4-kv", rendered)
        self.assertIn("--kv-disk-space-mb 8192", rendered)

    def test_ds4_dspark_256k_is_a_separate_high_context_candidate(self) -> None:
        candidate = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-256k"
        ]
        self.assertEqual(candidate["context"], 262_144)
        self.assertEqual(candidate["risk"], "experimental")
        self.assertEqual(candidate["features"], ["dspark", "kv-cache"])
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, candidate)
        )
        self.assertIn("--ctx 262144 --prefill-chunk 1024", rendered)
        self.assertIn("--dspark --dspark-confidence 0.7", rendered)
        self.assertIn("--kv-disk-dir /var/cache/ds4-kv", rendered)

    def test_ds4_think_max_profile_clears_exact_runtime_threshold(self) -> None:
        profile = self.catalog.profiles[
            "ds4-deepseek-v4-flash-hybrid-dspark-384k-think-max"
        ]
        self.assertEqual(profile["context"], 393_216)
        self.assertEqual(profile["risk"], "experimental")
        self.assertEqual(profile["features"], ["dspark", "kv-cache"])
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, profile)
        )
        self.assertIn("--ctx 393216 --prefill-chunk 1024", rendered)
        self.assertIn("--dspark --dspark-confidence 0.7", rendered)
        self.assertIn("--kv-disk-dir /var/cache/ds4-kv", rendered)

    def test_ds4_recipe_pins_archive_base_and_fixed_source(self) -> None:
        recipe = (ROOT / "lib/halo_ai/ds4/Containerfile").read_text(encoding="utf-8")
        self.assertIn(cli.DS4_RELEASE_SHA256, recipe)
        self.assertIn(str(cli.DS4_RELEASE_BYTES), recipe.replace("_", ""))
        self.assertIn(cli.DS4_UBUNTU_BASE, recipe)
        self.assertIn(cli.DS4_SOURCE_COMMIT, recipe)
        self.assertNotIn("git checkout", recipe)

    def test_ds4_image_requires_every_provenance_label(self) -> None:
        labels = {
            "local.halo-ai.engine": "ds4",
            "local.halo-ai.engine-archive-sha256": cli.DS4_RELEASE_SHA256,
            "org.opencontainers.image.revision": cli.DS4_SOURCE_COMMIT,
            "local.halo-ai.rocm-version": cli.DS4_ROCM_VERSION,
            "local.halo-ai.base": cli.DS4_UBUNTU_BASE,
        }
        with mock.patch.object(cli, "image_labels", return_value=labels):
            self.assertTrue(cli.ds4_image_valid("fixture"))
        labels.pop("org.opencontainers.image.revision")
        with mock.patch.object(cli, "image_labels", return_value=labels):
            self.assertFalse(cli.ds4_image_valid("fixture"))

    def test_ds4_backend_provenance_requires_active_dspark_process(self) -> None:
        commit = mock.MagicMock(stdout=cli.DS4_SOURCE_COMMIT + "\n")
        process = mock.MagicMock(
            stdout=(
                "COMMAND\n/opt/ds4/ds4-server --rocm --mtp /models/dspark.gguf "
                "--dspark --dspark-confidence 0.7\n"
            )
        )
        labels = {
            "org.opencontainers.image.version": "b0001",
            "local.halo-ai.rocm-version": cli.DS4_ROCM_VERSION,
            "local.halo-ai.engine-archive-sha256": cli.DS4_RELEASE_SHA256,
        }
        with (
            mock.patch.object(cli, "ds4_image_valid", return_value=True),
            mock.patch.object(cli, "podman", side_effect=[commit, process]),
            mock.patch.object(cli, "image_labels", return_value=labels),
        ):
            info = cli.ds4_backend_info(
                "halo-ds4", "fixture", {"features": ["dspark"]},
            )
        self.assertEqual(info["source_revision"], cli.DS4_SOURCE_COMMIT)
        self.assertEqual(info["speculation"], "dspark")

    def test_rocmfpx_strict_mtp_rejects_incompatible_ngram_composition(self) -> None:
        profile = json.loads(json.dumps(
            self.catalog.profiles["qwen3.8-27b-rocmfp4-mtp"]
        ))
        profile["features"].append("ngram")
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog(self.catalog.models, {profile["id"]: profile})

    def test_rocmfpx_fp8_profiles_are_explicit_gpu_only_artifacts(self) -> None:
        model = self.catalog.models["qwen3.8-27b-rocmfp8"]
        self.assertEqual(model["quantization"], "Q8_0_ROCMFPX")
        self.assertEqual(model["engines"], ["rocmfpx"])
        self.assertEqual(model["modalities"], ["text"])
        self.assertEqual(model["files"][0]["bytes"], 28_193_396_704)
        self.assertEqual(
            model["files"][0]["sha256"],
            "0bf5bfc9f946090af2d41b388ccb4d627e916c7250517c36a0de37d6eaccfd8e",
        )
        self.assertEqual(model["gguf_expectations"]["metadata"]["general.file_type"], 111)
        self.assertEqual(
            model["gguf_expectations"]["tensor_type_counts"], {"0": 360, "103": 506},
        )

        baseline = self.catalog.profiles["qwen3.8-27b-rocmfp8-baseline"]
        rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, baseline)
        )
        self.assertIn("Qwen3.8-27B-ROCmFP8.gguf,ro", rendered)
        self.assertIn("--spec-type none", rendered)
        self.assertNotIn("draft-mtp", rendered)
        with mock.patch.object(cli, "rocmfpx_image_valid", return_value=True):
            plan = cli.profile_acquisition_plan(self.config, self.catalog, baseline)
        self.assertEqual(plan["selected_artifact_classes"], ["fp8", "rocmfpx-runtime"])
        self.assertEqual(
            plan["excluded_artifact_classes"],
            ["fp4", "npu", "vision", "bf16", "reference"],
        )
        self.assertEqual(plan["model"]["files"][0]["bytes"], 28_193_396_704)

        mtp = self.catalog.profiles["qwen3.8-27b-rocmfp8-mtp"]
        mtp_rendered = __import__("shlex").join(
            cli.render_container(self.config, self.catalog, mtp)
        )
        self.assertIn("--spec-type draft-mtp", mtp_rendered)
        self.assertIn("--spec-mtp-strict-qwen", mtp_rendered)
        self.assertNotIn("--device=/dev/kfd", mtp_rendered)

    def test_rocmfpx_recipe_pins_archive_and_bases(self) -> None:
        recipe = (ROOT / "lib/halo_ai/rocmfpx/Containerfile").read_text(encoding="utf-8")
        self.assertIn(cli.ROCMFPX_ENGINE_SHA256, recipe)
        self.assertIn(cli.ROCMFPX_VULKAN_BASE, recipe)
        self.assertIn(cli.ROCMFPX_ROCM_BASE, recipe)
        self.assertIn(cli.ROCMFPX_Q38ROCM_COMMIT, recipe)
        self.assertNotIn("git checkout", recipe)

    def test_rocmfpx_image_requires_every_provenance_label(self) -> None:
        labels = {
            "local.halo-ai.engine": "rocmfpx",
            "local.halo-ai.engine-archive-sha256": cli.ROCMFPX_ENGINE_SHA256,
            "org.opencontainers.image.revision": cli.ROCMFPX_Q38ROCM_COMMIT,
            "local.halo-ai.vulkan-base": cli.ROCMFPX_VULKAN_BASE,
            "local.halo-ai.rocm-base": cli.ROCMFPX_ROCM_BASE,
        }
        with mock.patch.object(cli, "image_labels", return_value=labels):
            self.assertTrue(cli.rocmfpx_image_valid("fixture"))
        labels.pop("local.halo-ai.rocm-base")
        with mock.patch.object(cli, "image_labels", return_value=labels):
            self.assertFalse(cli.rocmfpx_image_valid("fixture"))


class ModelTests(unittest.TestCase):
    class Response:
        def __init__(self, payload: bytes, status: int = 200) -> None:
            self.payload = payload
            self.status = status

        def read(self, _size: int = -1) -> bytes:
            payload, self.payload = self.payload, b""
            return payload

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def test_scanner_ignores_appledouble_and_classifies_companions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            model_root = config.path("HALO_AI_MODELS_ROOT")
            model_root.mkdir(parents=True)
            (model_root / "main.gguf").write_bytes(gguf_fixture(b"payload"))
            (model_root / "mmproj-F32.gguf").write_bytes(gguf_fixture(b"projector"))
            (model_root / "dspark-model.gguf").write_bytes(gguf_fixture(b"draft"))
            (model_root / "._main.gguf").write_bytes(b"not a model")
            found = cli.scan_models(config)
            self.assertEqual(sorted(item["role"] for item in found), ["dspark", "main", "mmproj"])
            self.assertNotIn("._main.gguf", {item["path"] for item in found})

    def test_fast_verification_checks_magic_and_exact_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            model_root = config.path("HALO_AI_MODELS_ROOT")
            model_root.mkdir(parents=True)
            payload = gguf_fixture(b"fixture")
            (model_root / "fixture.gguf").write_bytes(payload)
            model = {
                "id": "fixture",
                "files": [
                    {
                        "role": "main",
                        "path": "fixture.gguf",
                        "bytes": len(payload),
                        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                    }
                ],
            }
            self.assertTrue(cli.verify_model(config, model, full=True)["valid"])

    def test_transformers_verification_checks_json_safetensors_and_processor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_config(root)
            model_root = config.path("HALO_AI_MODELS_ROOT")
            model_root.mkdir(parents=True)
            files = {
                "fixture/config.json": b"{}",
                "fixture/model.safetensors": (8).to_bytes(8, "little") + b'{"x":{}}' + b"data",
                "fixture/tokenizer.model": b"processor",
            }
            entries = []
            roles = ("main", "weights", "processor")
            for (relative, payload), role in zip(files.items(), roles):
                path = model_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                entries.append({
                    "role": role,
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                })
            model = {"id": "fixture", "format": "transformers", "files": entries}
            self.assertTrue(cli.verify_model(config, model, full=True)["valid"])

    def test_pinned_download_resumes_then_atomically_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            payload = b"verified fixture"
            entry = {
                "role": "main", "path": "owner/repo/model.gguf", "bytes": len(payload),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            }
            destination = config.path("HALO_AI_MODELS_ROOT") / entry["path"]
            destination.parent.mkdir(parents=True)
            partial = destination.with_name(f".{destination.name}.partial")
            partial.write_bytes(payload[:8])
            requests = []

            def open_fixture(request: object, timeout: int) -> object:
                requests.append((request, timeout))
                return self.Response(payload[8:], 206)

            with mock.patch.object(cli.urllib.request, "urlopen", side_effect=open_fixture):
                cli.download_huggingface_file(
                    config, "owner/repo", "a" * 40, entry, destination,
                )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(requests[0][0].get_header("Range"), "bytes=8-")
            self.assertEqual(requests[0][1], 120)

    def test_pinned_download_refuses_symlink_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            payload = b"fixture"
            entry = {
                "role": "main", "path": "owner/repo/model.gguf", "bytes": len(payload),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            }
            destination = config.path("HALO_AI_MODELS_ROOT") / entry["path"]
            destination.parent.mkdir(parents=True)
            outside = Path(temporary) / "outside"
            outside.write_bytes(b"unchanged")
            destination.with_name(f".{destination.name}.lock").symlink_to(outside)
            with self.assertRaises(cli.HaloError):
                cli.download_huggingface_file(
                    config, "owner/repo", "a" * 40, entry, destination,
                )
            self.assertEqual(outside.read_bytes(), b"unchanged")

    def test_gguf_semantic_inventory_is_fail_closed(self) -> None:
        observed = {
            "version": 3, "tensor_count": 866, "metadata_count": 51,
            "metadata": {
                "general.architecture": "qwen35",
                "tokenizer.ggml.tokens": {"subtype": 8, "length": 248320},
            },
            "tensor_type_counts": {"101": 505},
            "tensor_names": ["token_embd.weight", "blk.64.nextn.eh_proj.weight"],
        }
        expected = {
            "version": 3, "tensor_count": 866, "metadata_count": 51,
            "metadata": {"general.architecture": "qwen35"},
            "array_lengths": {"tokenizer.ggml.tokens": 248320},
            "tensor_type_counts": {"101": 505},
            "required_tensors": ["token_embd.weight", "blk.64.nextn.eh_proj.weight"],
            "forbidden_tensor_terms": ["vision"],
        }
        with mock.patch.object(cli, "gguf_inventory", return_value=observed):
            self.assertTrue(cli.verify_gguf_expectations(Path("fixture"), expected)["valid"])
            observed["tensor_names"].append("vision.patch.weight")
            result = cli.verify_gguf_expectations(Path("fixture"), expected)
        self.assertFalse(result["valid"])
        self.assertEqual(result["forbidden_tensor_terms_found"], ["vision"])

    def test_full_verification_inventory_preserves_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            first = {"model": "first", "valid": True, "full": True, "files": []}
            second = {"model": "second", "valid": True, "full": True, "files": []}
            replacement = {"model": "first", "valid": False, "full": True, "files": []}
            cli.record_full_verifications(config, [first, second])
            cli.record_full_verifications(config, [replacement])
            inventory = json.loads(
                config.path("HALO_AI_INVENTORY_FILE").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["model"] for item in inventory["results"]],
                ["first", "second"],
            )
            self.assertFalse(inventory["results"][0]["valid"])


class TrialTests(unittest.TestCase):
    def test_exited_container_has_no_live_cgroup_oom_counter(self) -> None:
        result = mock.MagicMock(returncode=0, stdout="0\n")
        with mock.patch.object(cli, "podman", return_value=result):
            self.assertIsNone(cli.container_oom_kill_count("fixture"))

    def test_start_failure_preserves_backend_error_for_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            logs = mock.MagicMock(
                stdout="0.03 E srv load_model: strict rollback is unavailable\n",
                stderr="",
            )
            trial: dict[str, object] = {}
            with (
                mock.patch.object(cli, "podman", return_value=logs),
                mock.patch.object(cli, "container_oom_kill_count", return_value=None),
            ):
                cli.record_start_failure(config, trial, "fixture", "readiness failed")
            self.assertIn("strict rollback is unavailable", trial["backend_error"])
            self.assertIn("load_model", trial["log_excerpt"])

    def test_rocmfpx_smoke_forces_deterministic_nonthinking_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            catalog = cli.load_catalog(config)
            models = {"data": [{"id": "Qwen3.8-27B-ROCmFP4-FAST.gguf"}]}
            completion = {
                "choices": [{"message": {"content": "halo-ai smoke test passed"}}],
                "system_fingerprint": "b213-e87d53e",
            }
            with (
                mock.patch.object(cli, "hardware_snapshot", return_value={}),
                mock.patch.object(cli, "http_json", side_effect=[models, completion]) as request,
            ):
                result = cli.command_test(
                    config, catalog,
                    __import__("argparse").Namespace(
                        profile_id="qwen3.8-27b-rocmfp4-baseline", preset=None,
                    ),
                )
            self.assertEqual(result, 0)
            payload = request.call_args_list[1].kwargs["payload"]
            self.assertEqual(payload["reasoning_effort"], "none")
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["seed"], 1)

    def test_rocmfpx_backend_provenance_requires_unassisted_vulkan_process(self) -> None:
        version = mock.MagicMock(stdout="version: 213 (e87d53e)\n", stderr="")
        process = mock.MagicMock(
            stdout="COMMAND\n/opt/rocmfpx/bin/llama-server --device Vulkan0 --spec-type none\n"
        )
        labels = {
            "org.opencontainers.image.revision": cli.ROCMFPX_Q38ROCM_COMMIT,
            "local.halo-ai.engine-archive-sha256": cli.ROCMFPX_ENGINE_SHA256,
            "local.halo-ai.vulkan-base": cli.ROCMFPX_VULKAN_BASE,
            "local.halo-ai.rocm-base": cli.ROCMFPX_ROCM_BASE,
        }
        with (
            mock.patch.object(cli, "podman", side_effect=[version, process]),
            mock.patch.object(cli, "image_labels", return_value=labels),
            mock.patch.object(cli, "rocmfpx_image_valid", return_value=True),
        ):
            info = cli.rocmfpx_backend_info(
                "halo-rocmfpx", "fixture-image", {"features": []},
            )
        self.assertEqual(info["device"], "Vulkan0")
        self.assertEqual(info["speculation"], "none")
        self.assertEqual(info["reported_source_revision"], "e87d53e-unresolved")

    def test_container_http_json_streams_payload_on_stdin(self) -> None:
        completed = mock.MagicMock(stdout='{"tokens":[1,2]}')
        with mock.patch.object(cli, "podman", return_value=completed) as podman:
            response = cli.container_http_json(
                "fixture", "http://127.0.0.1:8000/completion",
                {"prompt": [10, 20]},
            )
        self.assertEqual(response["tokens"], [1, 2])
        arguments = podman.call_args.args[0]
        self.assertIn("-i", arguments)
        self.assertIn("@-", arguments)
        self.assertNotIn("10", arguments)
        self.assertIn("10", podman.call_args.kwargs["input_text"])

    def test_lemonade_config_accepts_completed_hot_swap_disconnect(self) -> None:
        failed = mock.MagicMock(returncode=1, stdout="", stderr="connection closed")
        passed = mock.MagicMock(returncode=0, stdout="updated", stderr="")
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(cli, "podman", side_effect=[failed, passed]) as podman,
            mock.patch.object(cli, "wait_http") as wait_http,
        ):
            cli.configure_lemonade_runtime(
                make_config(Path(temporary)), "halo-lemonade", "http://127.0.0.1/live",
            )
        self.assertEqual(podman.call_count, 2)
        wait_http.assert_called_once_with("http://127.0.0.1/live", 180)

    def test_lemonade_resolver_accepts_exact_multimodal_folder_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            catalog = cli.load_catalog(config)
            profile = catalog.profiles["qwen3.6-27b-q8xl-vision-lemonade"]
            main = config.path("HALO_AI_MODELS_ROOT") / catalog.models[profile["model"]]["files"][0]["path"]
            main.parent.mkdir(parents=True)
            main.write_bytes(b"GGUF")
            response = {"data": [{
                "id": "qwen3.6-27b-q8xl-vision",
                "source": "extra_models_dir",
                "checkpoint": "/models/extra/qwen3.6-27b-q8xl-vision",
                "checkpoints": {
                    "main": "/models/extra/qwen3.6-27b-q8xl-vision",
                    "mmproj": "mmproj-F32.gguf",
                },
                "labels": ["custom", "vision", "mtp"],
            }]}
            with mock.patch.object(cli, "http_json", return_value=response):
                self.assertEqual(cli.exact_lemonade_model(config, catalog, profile, 13305), "qwen3.6-27b-q8xl-vision")

    def test_lemonade_text_identity_never_mounts_mmproj(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            catalog = cli.load_catalog(config)
            model = catalog.models["qwen3.6-27b-q8xl"]
            for entry in model["files"]:
                path = config.path("HALO_AI_MODELS_ROOT") / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"GGUF" if path.suffix == ".gguf" else b"template")
            mounts = cli.lemonade_runtime_mounts(config, catalog)
            self.assertNotIn("/models/extra/qwen3.6-27b-q8xl/mmproj-F32.gguf", mounts)
            self.assertIn("/models/extra/qwen3.6-27b-q8xl-vision/mmproj-F32.gguf", mounts)


class LongBenchTests(unittest.TestCase):
    @staticmethod
    def item(identifier: str, difficulty: str = "easy", length: str = "short") -> dict[str, str]:
        return {
            "_id": identifier, "domain": "Test", "sub_domain": "Fixture",
            "difficulty": difficulty, "length": length, "question": "Which?",
            "choice_A": "Alpha", "choice_B": "Beta", "choice_C": "Gamma",
            "choice_D": "Delta", "answer": "B", "context": "fixture context",
        }

    def test_prompt_and_answer_extraction_match_official_zero_shot_contract(self) -> None:
        prompt = cli.longbench.render_prompt(self.item("one"))
        self.assertIn("<text>\nfixture context\n</text>", prompt)
        self.assertIn("(B) Beta", prompt)
        self.assertEqual(
            cli.longbench.extract_answer("The correct answer is (B)"),
            "B",
        )
        self.assertIsNone(cli.longbench.extract_answer("I think it might be B"))

    def test_canary_is_one_stable_sample_per_difficulty_length_cell(self) -> None:
        items = [
            self.item("z", "easy", "short"), self.item("a", "easy", "short"),
            self.item("b", "hard", "long"), self.item("c", "hard", "medium"),
        ]
        selected = cli.longbench.select_items(items, suite="canary")
        self.assertEqual([item["_id"] for item in selected], ["a", "b", "c"])

    def test_exact_sample_selection_bypasses_canary_cells(self) -> None:
        items = [self.item("one"), self.item("two", "hard", "long")]
        selected = cli.longbench.select_items(items, suite="canary", sample_id="two")
        self.assertEqual([item["_id"] for item in selected], ["two"])
        with self.assertRaises(cli.longbench.LongBenchError):
            cli.longbench.select_items(items, suite="full", sample_id="missing")

    def test_middle_truncation_uses_measured_budget_and_preserves_both_ends(self) -> None:
        item = self.item("one")
        item["context"] = "LEFT" + "x" * 1000 + "RIGHT"
        prompt, count = cli.longbench.truncate_to_budget(item, 500, lambda value: len(value))
        self.assertLessEqual(count, 500)
        self.assertIn("LEFT", prompt)
        self.assertIn("RIGHT", prompt)
        self.assertIn("middle truncated by halo-ai", prompt)

    def test_score_reports_coverage_and_never_counts_skips_as_completed(self) -> None:
        score = cli.longbench.score_records([
            {"status": "complete", "judge": True, "difficulty": "easy", "length": "short", "truncated": False},
            {"status": "complete", "judge": False, "difficulty": "hard", "length": "long", "truncated": True},
            {"status": "skipped_overflow", "difficulty": "hard", "length": "long"},
        ])
        self.assertEqual(score["completed"], 2)
        self.assertEqual(score["skipped_overflow"], 1)
        self.assertEqual(score["truncated"], 1)
        self.assertEqual(score["accuracy_percent"], 50.0)

    def test_lemonade_token_counter_streams_request_on_stdin(self) -> None:
        processes = mock.MagicMock()
        processes.stdout = "COMMAND\n/opt/lemonade/llama-server --port 8001\n"
        count = mock.MagicMock()
        count.stdout = '{"object":"response.input_tokens","input_tokens":42}'
        with mock.patch.object(cli, "podman", side_effect=[processes, count]) as podman:
            self.assertEqual(
                cli.longbench_input_tokens(
                    "http://127.0.0.1:13305", "fixture", "large prompt",
                    lemonade_container="halo-lemonade",
                ),
                42,
            )
        arguments = podman.call_args_list[1].args[0]
        keywords = podman.call_args_list[1].kwargs
        self.assertIn("-i", arguments)
        self.assertIn("@-", arguments)
        self.assertIn("large prompt", keywords["input_text"])
        self.assertNotIn("large prompt", arguments)

    def test_lemonade_payload_excludes_server_managed_arguments(self) -> None:
        profile = {
            "context": 32768,
            "features": [],
            "settings": {"batch": 2048, "ubatch": 512, "parallel": 1, "kv": "f16", "load_mode": "none"},
        }
        payload = cli.lemonade_load_payload(
            profile,
            "extra.fixture",
            "/models/templates/qwen3.6-nonthinking.jinja",
        )
        self.assertNotIn("--metrics", payload["llamacpp_args"].split())
        self.assertIn("--chat-template-file /models/templates/qwen3.6-nonthinking.jinja", payload["llamacpp_args"])
        self.assertIn("--load-mode none", payload["llamacpp_args"])
        self.assertIn("--spec-type none", payload["llamacpp_args"])
        self.assertEqual(payload["ctx_size"], 32768)


class RocmFpxTuningTests(unittest.TestCase):
    def test_fixed_quality_suite_is_valid_and_long_case_is_deterministic(self) -> None:
        suite_path = ROOT / "config/benchmarks/rocmfpx-quality-v1.json"
        suite, digest = cli.rocmfpx_quality.load_suite(suite_path)
        self.assertEqual(suite["suite_id"], "halo-ai-rocmfpx-quality-v1")
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(suite["cases"]), 13)
        long_case = next(case for case in suite["cases"] if case["id"] == "long-context-needle")
        first = cli.rocmfpx_quality.case_messages(suite, long_case)
        second = cli.rocmfpx_quality.case_messages(suite, long_case)
        self.assertEqual(first, second)
        self.assertIn("Archive record 000397", first[-1]["content"])
        self.assertEqual(first[-1]["content"].count("quartz-8142"), 1)

    def test_quality_scoring_handles_exact_and_semantic_json(self) -> None:
        exact = {"id": "exact", "validator": {"type": "exact", "expected": "42"}}
        structured = {
            "id": "json", "validator": {
                "type": "json", "expected": {"city": "Kyiv", "days": 3},
            },
        }
        self.assertTrue(cli.rocmfpx_quality.score_case(exact, " 42\n")[0])
        self.assertTrue(cli.rocmfpx_quality.score_case(
            structured, '```json\n{"days":3,"city":"Kyiv"}\n```',
        )[0])

    def test_quality_comparison_proves_only_cross_process_same_model_identity(self) -> None:
        def record(profile: str, features: list[str], model_sha: str, token: str) -> dict[str, object]:
            return {
                "kind": "halo-ai-rocmfpx-quality-benchmark",
                "profile": profile,
                "features": features,
                "suite_sha256": "suite",
                "model": {"sha256": model_sha, "quantization": "fixture"},
                "results": [
                    {"case_id": "one", "category": "code", "passed": True, "output_token_sha256": token},
                    {"case_id": "two", "category": "reasoning", "passed": True, "output_token_sha256": token + "2"},
                ],
            }

        records = [
            record("fp4-baseline", [], "fp4", "same"),
            record("fp4-baseline", [], "fp4", "same"),
            record("fp4-mtp", ["mtp"], "fp4", "same"),
            record("fp4-mtp", ["mtp"], "fp4", "same"),
            record("fp8-baseline", [], "fp8", "different"),
            record("fp8-baseline", [], "fp8", "different"),
        ]
        result = cli.rocmfpx_quality.compare_records(records)
        self.assertTrue(result["profiles"]["fp4-baseline"]["self_consistent"])
        self.assertEqual(result["strict_identity"]["fp4-mtp"]["status"], "proven")
        self.assertNotIn("fp8-baseline", result["strict_identity"])

    def test_unique_context_text_is_deterministic_and_line_variant(self) -> None:
        first = cli.rocmfpx_unique_benchmark_text(3)
        self.assertEqual(first, cli.rocmfpx_unique_benchmark_text(3))
        self.assertIn("Record 000000", first)
        self.assertIn("Record 000002", first)
        self.assertEqual(first.count("\n"), 3)

    def test_context_benchmark_summary_uses_medians_and_peak_memory(self) -> None:
        runs = [
            {
                "prompt_tokens": 4095,
                "ttft_seconds": 10.0,
                "prompt_tokens_per_second": 100.0,
                "decode_tokens_per_second": 10.0,
                "peak_gtt_bytes": 20,
                "peak_vram_bytes": 4,
                "drafted_tokens": 10,
                "accepted_tokens": 8,
            },
            {
                "prompt_tokens": 4095,
                "ttft_seconds": 12.0,
                "prompt_tokens_per_second": 120.0,
                "decode_tokens_per_second": 12.0,
                "peak_gtt_bytes": 30,
                "peak_vram_bytes": 5,
                "drafted_tokens": 10,
                "accepted_tokens": 10,
            },
        ]
        result = cli.rocmfpx_tune.summarize_context_runs(runs)
        self.assertEqual(result[0]["ttft_seconds_median"], 11.0)
        self.assertEqual(result[0]["prompt_tokens_per_second_median"], 110.0)
        self.assertEqual(result[0]["decode_tokens_per_second_median"], 11.0)
        self.assertEqual(result[0]["peak_gtt_bytes_max"], 30)
        self.assertEqual(result[0]["acceptance_percent"], 90.0)

    def test_context_comparison_reports_end_to_end_crossover(self) -> None:
        def record(profile: str, ttft: float, pp: float, tps: float) -> dict[str, object]:
            return {
                "kind": "halo-ai-rocmfpx-context-benchmark",
                "profile": profile,
                "runs": [{
                    "prompt_tokens": 4095, "completion_tokens": 64,
                    "repetition": 1, "prompt_sha256": "same",
                }],
                "summary": [{
                    "prompt_tokens": 4095,
                    "ttft_seconds_median": ttft,
                    "prompt_tokens_per_second_median": pp,
                    "decode_tokens_per_second_median": tps,
                    "peak_gtt_bytes_max": 100,
                    "acceptance_percent": 90.0,
                }],
            }

        result = cli.rocmfpx_tune.compare_context_records(
            record("baseline", 20.0, 200.0, 10.0),
            record("candidate", 22.0, 180.0, 20.0),
        )
        context = result["contexts"][0]
        self.assertEqual(context["candidate_faster_after_generated_tokens"], 40)
        self.assertEqual(context["decode_speed_change_percent"], 100.0)
        self.assertTrue(result["identical_prompt_tokens"])

    def test_recorded_mtp_result_is_faster_but_held_experimental(self) -> None:
        baseline = json.loads((
            ROOT / "docs/results/qwen3.8-rocmfp4-baseline-2026-08-16.json"
        ).read_text(encoding="utf-8"))
        candidate = json.loads((
            ROOT / "docs/results/qwen3.8-rocmfp4-mtp-2026-08-16.json"
        ).read_text(encoding="utf-8"))
        result = cli.rocmfpx_tune.compare_mtp_records(baseline, candidate)
        self.assertEqual(result["decision"], "hold-experimental")
        self.assertTrue(result["evidence"]["same_model_sha256"])
        self.assertTrue(result["evidence"]["decode_at_least_10_percent_faster_every_context"])
        self.assertFalse(result["evidence"]["strict_token_identity_proven"])
        self.assertEqual(
            [row["decode_speed_change_percent"] for row in result["contexts"]],
            [29.51, 59.08, 45.92],
        )

    def test_mtp_comparison_rejects_different_prompt_sets(self) -> None:
        baseline = {
            "server_benchmarks": [{"prompt_tokens": 64}],
        }
        candidate = {"benchmarks": [{"prompt_tokens": 4096}]}
        with self.assertRaises(cli.rocmfpx_tune.TuneError):
            cli.rocmfpx_tune.compare_mtp_records(baseline, candidate)

    def test_mtp_payload_replaces_baseline_speculation_mode(self) -> None:
        profile = {
            "context": 32768,
            "features": ["mtp"],
            "settings": {
                "batch": 2048, "ubatch": 512, "parallel": 1, "kv": "f16",
                "load_mode": "none", "spec_draft_n_max": 2,
            },
        }
        args = cli.lemonade_load_payload(profile)["llamacpp_args"]
        self.assertIn("--spec-type draft-mtp", args)
        self.assertNotIn("--spec-type none", args)

    def test_cgroup_oom_stages_non_mmap_future_trial_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            trial = {
                "profile": "qwen3.6-27b-q8xl-lemonade",
                "boot_id": "fixture-boot",
                "features": [],
                "settings": {"load_mode": "mmap"},
                "context": 32768,
            }
            adjustment = cli.stage_pressure_reduction(config, trial)
            self.assertEqual(adjustment["HALO_AI_PENDING_LOAD_MODE"], "none")

    def test_wait_http_retries_connection_reset_during_restart(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        with (
            mock.patch.object(
                cli.urllib.request,
                "urlopen",
                side_effect=[ConnectionResetError(104, "Connection reset by peer"), response],
            ) as urlopen,
            mock.patch.object(cli.time, "sleep"),
        ):
            cli.wait_http("http://127.0.0.1:13305/live", seconds=2)
        self.assertEqual(urlopen.call_count, 2)

    def test_nonthinking_smoke_rejects_reasoning_only_response(self) -> None:
        response = {
            "choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]
        }
        with self.assertRaises(cli.HaloError):
            cli.validate_smoke_response(response, "halo-ai smoke test passed", False)

    def test_nonthinking_smoke_accepts_exact_content(self) -> None:
        response = {
            "choices": [{"message": {"content": "halo-ai smoke test passed"}}]
        }
        cli.validate_smoke_response(response, "halo-ai smoke test passed", False)

    def test_thinking_smoke_accepts_exact_content_without_forcing_a_trace(self) -> None:
        response = {
            "choices": [{"message": {"content": "halo-ai smoke test passed"}}]
        }
        cli.validate_smoke_response(response, "halo-ai smoke test passed", True)

    def test_deepseek_reasoning_content_is_valid_in_native_thinking_mode(self) -> None:
        response = {
            "choices": [{"message": {
                "content": "halo-ai smoke test passed",
                "reasoning_content": "internal trace",
            }}]
        }
        cli.validate_smoke_response(response, "halo-ai smoke test passed", True)

    def test_reasoning_template_canary_distinguishes_modes(self) -> None:
        top = mock.MagicMock()
        top.stdout = (
            "COMMAND\n/opt/lemonade/bin/llama-server --port 8001 -m /models/model.gguf\n"
        )
        disabled = mock.MagicMock()
        disabled.stdout = json.dumps({"prompt": "assistant\n<think>\n\n</think>\n\n"})
        enabled = mock.MagicMock()
        enabled.stdout = json.dumps({"prompt": "assistant\n"})
        with mock.patch.object(cli, "podman", side_effect=[top, disabled, top, enabled]):
            cli.validate_lemonade_reasoning_template(
                "halo-lemonade",
                {"messages": [], "reasoning_effort": "none"},
                False,
            )
            cli.validate_lemonade_reasoning_template(
                "halo-lemonade",
                {"messages": [], "chat_template_kwargs": {"enable_thinking": True}},
                True,
            )

    def test_thinking_preset_removes_inherited_reasoning_off_switch(self) -> None:
        payload = {"reasoning_effort": "none", "temperature": 0.7}
        cli.merge_request_preset(
            payload,
            {"chat_template_kwargs": {"enable_thinking": True}, "temperature": 0.6},
        )
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["temperature"], 0.6)

    def test_mtp_metrics_require_drafted_and_accepted_tokens(self) -> None:
        metrics = cli.validate_mtp_metrics({
            "timings": {"draft_n": 8, "draft_n_accepted": 6, "predicted_per_second": 50.0}
        })
        self.assertEqual(metrics["draft_n_accepted"], 6)
        with self.assertRaises(cli.HaloError):
            cli.validate_mtp_metrics({"timings": {"draft_n": 8, "draft_n_accepted": 0}})

    def test_dspark_metrics_require_accepted_tokens(self) -> None:
        metrics = cli.validate_speculative_metrics({
            "timings": {"draft_n": 10, "draft_n_accepted": 7, "predicted_per_second": 20.0}
        }, "dspark")
        self.assertEqual(metrics["draft_n_accepted"], 7)

    def test_ds4_log_timings_are_normalized_for_benchmark_records(self) -> None:
        result = mock.MagicMock()
        result.stdout = (
            "ds4-server: chat ctx=0..10278:10278 prompt done 105.162s\n"
            "ds4-server: chat ctx=10278..10304:26 gen=26 decoding "
            "chunk=12.56 t/s avg=12.56 t/s 2.071s\n"
        )
        with mock.patch.object(cli, "podman", return_value=result):
            timings = cli.ds4_recent_timings(
                "halo-ds4", {"prompt_tokens": 10278, "completion_tokens": 26}
            )
        self.assertIsNotNone(timings)
        self.assertAlmostEqual(timings["prompt_per_second"], 97.73, places=2)
        self.assertEqual(timings["predicted_per_second"], 12.56)
        self.assertEqual(timings["source"], "ds4_server_log")

    def test_ds4_cached_timings_report_only_newly_processed_prompt_tokens(self) -> None:
        logs = (
            "ds4-server: chat ctx=10240..10278:38 prompt done 1.519s\n"
            "ds4-server: chat ctx=10278..10304:26 gen=26 decoding "
            "chunk=12.83 t/s avg=12.83 t/s 2.026s\n"
        )
        completed = __import__("subprocess").CompletedProcess([], 0, stdout="", stderr=logs)
        with mock.patch.object(cli, "podman", return_value=completed):
            timings = cli.ds4_recent_timings("halo-ds4", {
                "prompt_tokens": 10278,
                "completion_tokens": 26,
                "prompt_tokens_details": {"cached_tokens": 10240},
            })
        self.assertEqual(timings["prompt_n"], 38)
        self.assertEqual(timings["prompt_cached_n"], 10240)
        self.assertAlmostEqual(timings["prompt_per_second"], 25.016, places=3)

    def test_lemonade_backend_versions_reports_package_and_active_binary(self) -> None:
        first = mock.MagicMock()
        first.stdout = (
            "llamacpp            cpu         installable     not installed\n"
            "                    rocm        installed       b10334\n"
            "moonshine           cpu         installable     not installed\n"
        )
        second = mock.MagicMock()
        second.stdout = (
            "COMMAND\n/opt/lemonade/.cache/lemonade/bin/llamacpp/rocm-stable/"
            "llama-b10333/llama-server -m /models/model.gguf\n"
        )
        with mock.patch.object(cli, "podman", side_effect=[first, second]):
            self.assertEqual(
                cli.lemonade_backend_versions("halo-lemonade"),
                {"package_version": "b10334", "binary_version": "b10333", "channel": "stable"},
            )

    def test_lemonade_backend_versions_handles_update_available_status(self) -> None:
        backends = mock.MagicMock()
        backends.stdout = (
            "llamacpp            cpu         installable     not installed\n"
            "                    rocm        update_availableNewer upstream release available: b10597\n"
            "moonshine           cpu         installable     not installed\n"
        )
        version = mock.MagicMock()
        version.stdout = "b10597\n"
        processes = mock.MagicMock()
        processes.stdout = (
            "COMMAND\n/opt/lemonade/.cache/lemonade/bin/llamacpp/rocm-stable/"
            "llama-b10594/llama-server -m /models/model.gguf\n"
        )
        with mock.patch.object(cli, "podman", side_effect=[backends, processes, version]):
            self.assertEqual(
                cli.lemonade_backend_versions("halo-lemonade"),
                {"package_version": "b10597", "binary_version": "b10594", "channel": "stable"},
            )

    def test_lemonade_backend_versions_handles_flat_nightly_binary(self) -> None:
        backends = mock.MagicMock()
        backends.stdout = (
            "llamacpp            cpu         installable     not installed\n"
            "                    rocm        update_availableNewer upstream release available: b1315\n"
            "moonshine           cpu         installable     not installed\n"
        )
        processes = mock.MagicMock()
        processes.stdout = (
            "COMMAND\n/opt/lemonade/.cache/lemonade/bin/llamacpp/rocm-nightly/"
            "llama-server -m /models/model.gguf\n"
        )
        version = mock.MagicMock()
        version.stdout = "b1315\n"
        binary_version = mock.MagicMock()
        binary_version.stdout = (
            "version: 0.2.0-dev (build 1, commit 95b8e33)\n"
            "built with Clang 23.0.0 for Linux\n"
        )
        with mock.patch.object(
            cli, "podman", side_effect=[backends, processes, version, binary_version],
        ):
            self.assertEqual(
                cli.lemonade_backend_versions("halo-lemonade"),
                {
                    "package_version": "b1315",
                    "channel": "nightly",
                    "binary_version": "b1",
                    "source_commit": "95b8e33",
                },
            )

    def test_lemonade_backend_running_uses_exact_process_name(self) -> None:
        result = mock.MagicMock(returncode=0)
        result.stdout = "COMMAND\nlemond\nllama-server\n"
        with mock.patch.object(cli, "podman", return_value=result):
            self.assertTrue(cli.lemonade_backend_running("halo-lemonade"))
        result.stdout = "COMMAND\nlemond\nds4-server\n"
        with mock.patch.object(cli, "podman", return_value=result):
            self.assertTrue(cli.lemonade_backend_running("halo-lemonade"))
        result.stdout = "COMMAND\nlemond\nmy-ds4-server-wrapper\n"
        with mock.patch.object(cli, "podman", return_value=result):
            self.assertFalse(cli.lemonade_backend_running("halo-lemonade"))

    def test_mtp_capability_requires_exact_model_label_and_backend_flags(self) -> None:
        top = mock.MagicMock()
        top.stdout = (
            "COMMAND\n/opt/lemonade/.cache/lemonade/bin/llamacpp/rocm-stable/"
            "llama-b10333/llama-server --port 8001\n"
        )
        help_result = mock.MagicMock()
        help_result.stdout = "--spec-type none,draft-mtp\n--spec-draft-n-max N\n"
        models = {"data": [{"id": "Qwen", "labels": ["custom", "mtp"]}]}
        with (
            mock.patch.object(cli, "http_json", return_value=models),
            mock.patch.object(cli, "podman", side_effect=[top, help_result]),
        ):
            cli.assert_lemonade_mtp_model(13305, "Qwen")
            cli.assert_lemonade_mtp_backend("halo-lemonade")

    def test_mtp_capability_rejects_unlabeled_model(self) -> None:
        models = {"data": [{"id": "Qwen", "labels": ["custom"]}]}
        with mock.patch.object(cli, "http_json", return_value=models):
            with self.assertRaises(cli.HaloError):
                cli.assert_lemonade_mtp_model(13305, "Qwen")

    def test_non_oom_failure_can_retry_in_same_boot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            catalog = cli.load_catalog(config)
            profile = catalog.profiles["qwen3.6-27b-q8xl-lemonade"]
            model = catalog.models[profile["model"]]
            cli.atomic_json(
                cli.state_path(config, "last-trial.json"),
                {
                    "status": "failed",
                    "confirmed_oom": False,
                    "boot_id": "fixture-boot",
                    "fingerprint": cli.trial_fingerprint(profile, model),
                },
            )
            with mock.patch.object(cli, "boot_id", return_value="fixture-boot"):
                cli.refuse_same_boot_retry(config, profile, model)

    def test_prior_boot_trial_is_suspected_lockup_not_confirmed_oom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            active = cli.state_path(config, "active-trial.json")
            cli.atomic_json(active, {"profile": "fixture", "boot_id": "old-boot", "status": "starting"})
            with mock.patch.object(cli, "boot_id", return_value="new-boot"):
                with self.assertRaises(cli.HaloError):
                    cli.detect_prior_lockup(config)
            history = json.loads(cli.state_path(config, "oom-history.jsonl").read_text().strip())
            self.assertEqual(history["classification"], "suspected_lockup")
            self.assertFalse(history["confirmed_oom"])

    def test_operator_stop_marker_matches_only_the_active_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            trial = {
                "boot_id": "fixture-boot",
                "engine": "lemonade",
                "fingerprint": "fixture-fingerprint",
                "profile": "fixture-profile",
                "started_at": "fixture-time",
            }
            cli.atomic_json(cli.state_path(config, "active-trial.json"), trial)
            cli.request_trial_stop(config, ["halo-lemonade"])
            self.assertTrue(cli.consume_trial_stop(config, trial))
            self.assertFalse(cli.state_path(config, "stop-request.json").exists())

    def test_stop_marker_does_not_cancel_a_different_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            trial = {
                "boot_id": "fixture-boot",
                "engine": "lemonade",
                "fingerprint": "first",
                "profile": "fixture-profile",
                "started_at": "fixture-time",
            }
            cli.atomic_json(cli.state_path(config, "active-trial.json"), trial)
            cli.request_trial_stop(config, ["halo-lemonade"])
            self.assertFalse(cli.consume_trial_stop(config, {**trial, "fingerprint": "second"}))
            self.assertTrue(cli.state_path(config, "stop-request.json").exists())

    def test_stale_stop_marker_is_removed_before_a_new_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            marker = cli.state_path(config, "stop-request.json")
            cli.atomic_json(marker, {"profile": "old-profile"})
            cli.detect_prior_lockup(config)
            self.assertFalse(marker.exists())

    def test_profile_is_disabled_when_weights_exceed_cpu_visible_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = make_config(Path(temporary))
            catalog = cli.load_catalog(config)
            profile = catalog.profiles["qwen3.6-27b-q8xl-lemonade"]
            with (
                mock.patch.object(cli, "verify_model", return_value={"valid": True}),
                mock.patch.object(cli, "meminfo_bytes", return_value=32 * 1024**3),
            ):
                ready, reason = cli.profile_availability(config, catalog, profile)
            self.assertFalse(ready)
            self.assertIn("unsafe memory topology", reason)


class HostProfileTests(unittest.TestCase):
    def test_npu_removes_only_iommu_tokens(self) -> None:
        original = 'quiet root=UUID=abc amd_iommu=off amdgpu.gttsize=114688 ttm.pages_limit=29360128'
        transformed = cli.host_profile.transform(original, "npu")
        self.assertEqual(transformed, 'quiet root=UUID=abc amdgpu.gttsize=114688 ttm.pages_limit=29360128')

    def test_gpu_removes_iommu_pt_and_adds_amd_off(self) -> None:
        original = 'quiet root=UUID=abc iommu=pt amdgpu.gttsize=114688'
        transformed = cli.host_profile.transform(original, "gpu")
        self.assertEqual(transformed, 'quiet root=UUID=abc amdgpu.gttsize=114688 amd_iommu=off')

    def test_gpu_profile_stages_exact_118_gib_pair(self) -> None:
        original = 'quiet root=UUID=abc amd_iommu=off amdgpu.gttsize=114688 ttm.pages_limit=29360128'
        transformed = cli.host_profile.transform(original, "gpu", 118)
        self.assertEqual(
            transformed,
            'quiet root=UUID=abc amd_iommu=off amdgpu.gttsize=120832 ttm.pages_limit=30932992',
        )
        self.assertEqual(cli.host_profile.gtt_from_cmdline(transformed), 118)

    def test_gtt_change_rejects_partial_pair(self) -> None:
        with self.assertRaises(cli.host_profile.HostProfileError):
            cli.host_profile.transform('quiet amdgpu.gttsize=114688', "gpu", 118)

    def test_duplicate_iommu_tokens_fail_closed(self) -> None:
        with self.assertRaises(cli.host_profile.HostProfileError):
            cli.host_profile.transform('quiet amd_iommu=off amd_iommu=off', "npu")

    def test_limine_render_preserves_non_cmdline_lines(self) -> None:
        text = 'ESP_PATH="/boot"\nKERNEL_CMDLINE[default]+="quiet amd_iommu=off root=UUID=abc"\nBOOT_ORDER="*"\n'
        rendered, _old, _new = cli.host_profile.render_defaults(text, "npu")
        self.assertEqual(rendered, 'ESP_PATH="/boot"\nKERNEL_CMDLINE[default]+="quiet root=UUID=abc"\nBOOT_ORDER="*"\n')

    def test_boot_backup_manifest_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            defaults = root / "etc/default/limine"
            defaults.parent.mkdir(parents=True)
            defaults.write_text('KERNEL_CMDLINE[default]+="quiet"\n', encoding="utf-8")
            boot = root / "boot"
            (boot / "EFI").mkdir(parents=True)
            (boot / "EFI/kernel.efi").write_bytes(b"kernel fixture")
            state = root / "state"
            with (
                mock.patch.object(cli.host_profile, "LIMINE_DEFAULTS", defaults),
                mock.patch.object(cli.host_profile, "BOOT_ROOT", boot),
                mock.patch.object(cli.host_profile, "STATE_ROOT", state),
                mock.patch.object(cli.host_profile, "BACKUP_ROOT", state / "backups"),
            ):
                manifest = cli.host_profile.create_backup("fixture")
                _directory, verified = cli.host_profile.verify_backup("fixture")
            self.assertEqual(manifest, verified)
            self.assertEqual(len(manifest["boot_files"]), 1)


if __name__ == "__main__":
    unittest.main()
