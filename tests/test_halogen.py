import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_cli import cli, make_config


class HalogenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = make_config(Path(self.tmp.name))
        self.catalog = cli.load_catalog(self.config)
        self.profile = self.catalog.profiles['qwen3.8-flash-next-halogen-262k-vision']

    def test_recommended_aliases_use_v2_without_enabling_downloads(self):
        for alias in ['qwen3.8-fn', 'qwen3.8-halogen', 'qwen3.8-halogen-npu']:
            with self.subTest(alias=alias), mock.patch.object(cli.halogen_npu, 'xrt_mounts', return_value=[]):
                profile = cli.resolve_profile(self.catalog, alias)
                self.assertEqual(profile['model'], 'halogen-qwen3.8-flash-next-v2')
                command = cli.render_container(self.config, self.catalog, profile)
                self.assertIn('HALOGEN_NGRAM_TABLE=/models/qwen38-flash-next-ngram.hgn', command)
                self.assertFalse(any('HALOGEN_DOWNLOAD' in value or 'HALOGEN_MTP_DEPTH' in value for value in command))

    def test_native_checkpoint_closure_excludes_external_head_and_speed_overlay(self):
        roles = cli.required_roles(self.profile)
        self.assertEqual(roles, {'main', 'overlay', 'processor', 'mmproj'})
        command = cli.render_container(self.config, self.catalog, self.profile)
        text = ' '.join(command)
        self.assertIn('HALOGEN_DRAFTER_DEFAULT=1', command)
        self.assertIn('HALOGEN_VISION_TOWER=/models/qwen38-flash-next-vision.hgn', command)
        self.assertNotIn('qwen38-flash-next-mtp.hgn', text)
        self.assertNotIn('overlay-speed', text)
        self.assertNotIn('HALOGEN_DOWNLOAD', text)
        self.assertIn('--ipc=host', command)
        self.assertIn('memlock=-1:-1', command)
        self.assertIn('127.0.0.1:8731:8731', command)
        self.assertNotIn('8730:8730', text)
        mounts = [command[i+1] for i, value in enumerate(command) if value == '--mount']
        self.assertTrue(all(value.endswith(',ro') for value in mounts))
        self.assertIn('/models/tokenizer/chat_template.jinja', text)
        self.assertEqual(command[-1], cli.HALOGEN_IMAGE)

    def test_baseline_disables_both_draft_sources_and_cache(self):
        profile = self.catalog.profiles['qwen3.8-flash-next-halogen-32k-baseline']
        command = cli.render_container(self.config, self.catalog, profile)
        for setting in ('HALOGEN_DRAFTER_DEFAULT=0', 'HALOGEN_PLD=0', 'HALOGEN_PROMPT_CACHE=0'):
            self.assertIn(setting, command)
        self.assertNotIn('mmproj', cli.required_roles(profile))

    def test_rejects_unsafe_memory_or_unknown_settings(self):
        for key, value in [('kv_pool_positions', 524288), ('prefill_chunk', 262144), ('parallel', True), ('extra_args', '--anything')]:
            with self.subTest(key=key):
                profile = copy.deepcopy(self.profile)
                profile['settings'][key] = value
                with self.assertRaises(cli.HaloError):
                    cli.validate_catalog(self.catalog.models, {profile['id']: profile})

    def test_hgn_header_and_hash_verification(self):
        root = self.config.path('HALO_AI_MODELS_ROOT')
        root.mkdir(parents=True)
        path = root / 'fixture.hgn'
        content = b'HGN1' + struct.pack('<IQ', 2, 3) + b'payload'
        path.write_bytes(content)
        model = {'id': 'fixture', 'format': 'hgn', 'files': [{'role': 'main', 'path': path.name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}]}
        self.assertTrue(cli.verify_model(self.config, model, True)['valid'])
        path.write_bytes(b'GGUF' + content[4:])
        self.assertFalse(cli.verify_model(self.config, model, False)['valid'])

    def test_image_provenance_and_health_use_engine_endpoint(self):
        with mock.patch.object(cli, 'image_identity', return_value='sha256:unexpected'):
            self.assertFalse(cli.halogen_image_valid('candidate'))
        with mock.patch.object(cli, 'image_identity', return_value=cli.HALOGEN_IMAGE_DIGEST):
            self.assertTrue(cli.halogen_image_valid('candidate'))
        self.assertEqual(cli.service_health_url(self.config, 'halogen'), 'http://127.0.0.1:8731/health')

    def test_nested_download_cannot_escape_repository(self):
        model = copy.deepcopy(self.catalog.models[self.profile['model']])
        entry = next(f for f in model['files'] if f['source_path'] == 'tokenizer/tokenizer.json')
        entry['path'] = 'another-repo/tokenizer.json'
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog({model['id']: model}, {})

    def test_health_rejects_wrong_version_or_silent_serial_fallback(self):
        health = {
            'context': 262144, 'slots': 1, 'kv_pool_positions': 262144,
            'drafter_default': 'mtp', 'checkpoint_format': 'hgn',
            'reasoning_effort_default': 'medium',
            'version': {'api': cli.HALOGEN_RELEASE, 'engine': cli.HALOGEN_RELEASE, 'match': True},
            'engine': {'responds': True}, 'drafter_weights_loaded': True,
            'capability_probe': 'ok',
            'chat_template': {'probe': 'passed'}, 'prompt_cache': {'mode': 2},
            'vision': {'enabled': True},
        }
        with mock.patch.object(cli, 'http_json', return_value=health):
            self.assertEqual(cli.halogen_backend_info(self.config, self.profile), health)
        for key, value in [('drafter_default', 'serial'), ('drafter_weights_loaded', False), ('version', {'api': 'old'}), ('vision', {'enabled': False}), ('capability_probe', 'fallback')]:
            with self.subTest(key=key), mock.patch.object(cli, 'http_json', return_value={**health, key: value}):
                with self.assertRaises(cli.HaloError):
                    cli.halogen_backend_info(self.config, self.profile)
        npu_profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-npu')
        names = npu_profile['settings']['npu_models']
        healthy_npu = {**health, 'npu': {'status': 'ok', 'models': names}}
        advertised = {'data': [{'id': name} for name in names]}
        with mock.patch.object(cli, 'http_json', side_effect=[healthy_npu, advertised]):
            self.assertEqual(cli.halogen_backend_info(self.config, npu_profile), healthy_npu)
        for npu in [{}, {'status': 'error', 'models': names}, {'status': 'ok', 'models': names[:1]}]:
            with mock.patch.object(cli, 'http_json', return_value={**health, 'npu': npu}):
                with self.assertRaises(cli.HaloError):
                    cli.halogen_backend_info(self.config, npu_profile)
        with mock.patch.object(cli, 'http_json', side_effect=[healthy_npu, {'data': []}]):
            with self.assertRaises(cli.HaloError):
                cli.halogen_backend_info(self.config, npu_profile)

    def test_v2_requires_external_table_and_never_mounts_w4b_overlay(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2')
        self.assertEqual(cli.required_roles(profile), {'main', 'processor', 'ngram', 'mmproj'})
        command = cli.render_container(self.config, self.catalog, profile)
        self.assertIn('HALOGEN_NGRAM_TABLE=/models/qwen38-flash-next-ngram.hgn', command)
        self.assertFalse(any('OVERLAY=' in value or 'w4b' in value for value in command))
        models = copy.deepcopy(self.catalog.models)
        models[profile['model']]['files'] = [f for f in models[profile['model']]['files'] if f['role'] != 'ngram']
        with self.assertRaises(cli.HaloError):
            cli.validate_catalog(models, {profile['id']: profile})

    def test_npu_is_explicit_and_does_not_enable_downloads_or_host_sysfs_writes(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-npu')
        with mock.patch.object(cli.halogen_npu, 'xrt_mounts', return_value=[(Path('/usr/lib/libxrt.so.2.21'), '/usr/lib/libxrt.so.2')]):
            command = cli.render_container(self.config, self.catalog, profile)
        self.assertIn('/dev/accel/accel0', command)
        self.assertIn('HALOGEN_NPU_MODELS=qwen3-embedding-0.6b,qwen3-reranker-0.6b', command)
        self.assertFalse(any('HALOGEN_DOWNLOAD' in s or '/host/sys' in s or 'NPU_WITH_GPU' in s for s in command))
        self.assertTrue(all(command[i+1].endswith(',ro') for i,s in enumerate(command) if s == '--mount'))
        for names in [[], ['unknown'], ['qwen3-reranker-0.6b'] * 2]:
            invalid = copy.deepcopy(profile)
            invalid['settings']['npu_models'] = names
            with self.assertRaises(cli.HaloError):
                cli.validate_catalog(self.catalog.models, {invalid['id']: invalid})

    def test_npu_reranker_includes_shared_program_without_embedder_weights(self):
        paths = [str(path) for path, _, _ in cli.halogen_npu.artifacts(Path('/models/npu'), ['qwen3-reranker-0.6b'])]
        self.assertTrue(any('qwen3-embedding-0.6b/devices/' in p for p in paths))
        self.assertFalse(any(p.endswith('qwen3-embedding-0.6b.hnpw') for p in paths))
        self.assertTrue(any(p.endswith('qwen3-reranker-0.6b.hnpw') for p in paths))

    def test_npu_fabric_clock_requires_only_the_top_level_selected(self):
        root = Path(self.tmp.name)
        device = root / 'sys/class/drm/card1/device'
        device.mkdir(parents=True)
        perf = device / 'power_dpm_force_performance_level'
        fclk = device / 'pp_dpm_fclk'
        accel = root / 'accel0'
        accel.touch()
        perf.write_text('manual\n')
        fclk.write_text('0: 800Mhz *\n1: 1800Mhz\n')
        with mock.patch.object(cli.halogen_npu, 'xrt_mounts', return_value=[]):
            self.assertTrue(cli.halogen_npu.host_errors(root, root / 'sys', accel))
            fclk.write_text('0: 800Mhz\n1: 1800Mhz *\n')
            self.assertEqual(cli.halogen_npu.host_errors(root, root / 'sys', accel), [])
            perf.write_text('auto\n')
            self.assertTrue(cli.halogen_npu.host_errors(root, root / 'sys', accel))

    def test_npu_host_check_catches_rootless_memlock_limit(self):
        with mock.patch.object(cli.halogen_npu.resource, 'getrlimit', return_value=(8388608, 8388608)):
            errors = cli.halogen_npu.host_errors(Path('/usr/lib'), required_memlock=4 * 1024**3)
            self.assertTrue(any('memlock limit' in error for error in errors))
        with mock.patch.object(cli.halogen_npu.resource, 'getrlimit', return_value=(-1, -1)):
            errors = cli.halogen_npu.host_errors(Path('/usr/lib'), required_memlock=4 * 1024**3)
            self.assertFalse(any('memlock limit' in error for error in errors))
