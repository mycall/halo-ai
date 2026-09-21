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
        self.profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen')

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
            'chat_template': {'probe': 'passed'}, 'prompt_cache': {'mode': 2},
            'vision': {'enabled': True},
        }
        with mock.patch.object(cli, 'http_json', return_value=health):
            self.assertEqual(cli.halogen_backend_info(self.config, self.profile), health)
        for key, value in [('drafter_default', 'serial'), ('drafter_weights_loaded', False), ('version', {'api': 'old'}), ('vision', {'enabled': False})]:
            with self.subTest(key=key), mock.patch.object(cli, 'http_json', return_value={**health, key: value}):
                with self.assertRaises(cli.HaloError):
                    cli.halogen_backend_info(self.config, self.profile)
