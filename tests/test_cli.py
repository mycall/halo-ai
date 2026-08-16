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
    def test_configuration_is_data_not_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "must-not-exist"
            config = Path(temporary) / "config.env"
            config.write_text(f"HALO_AI_RUN_USER=$(touch {marker})\n", encoding="utf-8")
            parsed = cli.parse_env_file(config)
            self.assertEqual(parsed["HALO_AI_RUN_USER"], f"$(touch {marker})")
            self.assertFalse(marker.exists())

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


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.config = make_config(Path(temporary))
            self.catalog = cli.load_catalog(self.config)

    def test_checked_in_catalog_validates(self) -> None:
        self.assertEqual(len(self.catalog.models), 5)
        self.assertEqual(len(self.catalog.profiles), 18)

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
            r"^([0-9a-f]{64}) ([0-9]+) ((?:antirez|unsloth|facebook)/[^\n]+\.(?:gguf|jinja|json|safetensors|model))$",
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


class ModelTests(unittest.TestCase):
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


class TrialTests(unittest.TestCase):
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
                {"package_version": "b10334", "binary_version": "b10333"},
            )

    def test_lemonade_backend_running_uses_exact_process_name(self) -> None:
        result = mock.MagicMock(returncode=0)
        result.stdout = "COMMAND\nlemond\nllama-server\n"
        with mock.patch.object(cli, "podman", return_value=result):
            self.assertTrue(cli.lemonade_backend_running("halo-lemonade"))

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
