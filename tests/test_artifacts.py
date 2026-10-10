"""Contracts between artifact selection, preview, acquisition and engine rendering."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from test_cli import cli, make_config
import artifact_store
import engine_halogen


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = make_config(Path(self.tmp.name))
        self.root = self.config.path('HALO_AI_MODELS_ROOT')
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = cli.load_catalog(self.config)

    @contextlib.contextmanager
    def installed_images(self):
        with contextlib.ExitStack() as stack:
            for engine in ('llamacpp', 'rocmfpx', 'strixvulkan', 'strixvulkan075', 'halogen', 'ds4'):
                stack.enter_context(mock.patch.object(cli, engine + '_image_valid', return_value=True))
            stack.enter_context(mock.patch.object(cli, 'image_identity', return_value='fixture'))
            yield

    def test_every_downloadable_profile_accounts_for_its_complete_selection(self):
        with self.installed_images():
            for profile in self.catalog.profiles.values():
                closure = cli.resolve_artifacts(self.config, self.catalog, profile)
                if any('download' not in s.model for s in closure.models):
                    continue
                with self.subTest(profile=profile['id']):
                    plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
                    groups = [plan['model'], plan['draft_model'], plan['auxiliary']]
                    files = [f for group in groups if group for f in group['files']]
                    expected = {str(path) for selection in closure.models for _, path in selection.files}
                    expected.update(str(a.destination) for a in closure.auxiliary)
                    self.assertEqual({f['destination'] for f in files}, expected)
                    self.assertEqual(len(files), len(expected))
                    self.assertEqual(plan['total_additional_download_bytes'], sum(f['bytes'] for f in files))
                    self.assertFalse(set(plan['selected_artifact_classes']) & set(plan['excluded_artifact_classes']))

    def test_halogen_variants_preview_and_mount_exactly_the_same_files(self):
        with self.installed_images(), mock.patch.object(cli.halogen_npu, 'xrt_mounts', return_value=[]):
            for alias in ('qwen3.8-fn', 'qwen3.8-halogen-v2', 'qwen3.8-halogen-npu', 'qwen3.8-halogen-v2-npu',
                          'qwen3.8-flash-next-halogen-262k-vision', 'qwen3.8-flash-next-halogen-262k-vision-npu'):
                with self.subTest(alias=alias):
                    profile = cli.resolve_profile(self.catalog, alias)
                    closure = cli.resolve_artifacts(self.config, self.catalog, profile)
                    plan = cli.profile_acquisition_plan(self.config, self.catalog, profile)
                    command = cli.render_container(self.config, self.catalog, profile)
                    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == '--mount']
                    mounted = {m.split('src=', 1)[1].split(',dst=', 1)[0] for m in mounts}
                    files = plan['model']['files'] + (plan['auxiliary']['files'] if plan['auxiliary'] else [])
                    self.assertEqual(mounted, {f['destination'] for f in files})
                    self.assertTrue(all(m.endswith(',ro') for m in mounts))
                    self.assertEqual('quality-overlay' in plan['selected_artifact_classes'], 'external-ngram' not in profile['features'])
                    self.assertEqual('external-ngram' in plan['selected_artifact_classes'], 'external-ngram' in profile['features'])
                    self.assertEqual('npu' in plan['selected_artifact_classes'], 'npu' in profile['features'])
                    if closure.auxiliary:
                        self.assertEqual(sum(a.bytes for a in closure.auxiliary), 1722419002)
                        self.assertEqual(engine_halogen.npu_memory_bytes(closure.auxiliary), 1722419002 + 2 * 1024**3)

    def test_npu_acquire_dry_run_never_downloads_or_installs(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2-npu')
        with self.installed_images(), contextlib.redirect_stdout(io.StringIO()) as output, \
                mock.patch.object(artifact_store, 'acquire') as acquire, \
                mock.patch.object(cli, 'command_runtime_install') as install:
            self.assertEqual(cli.command_profile_acquire(self.config, self.catalog, profile, dry_run=True), 0)
        acquire.assert_not_called()
        install.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertGreater(plan['auxiliary']['additional_download_bytes'], 0)
        self.assertIn('npu', plan['selected_artifact_classes'])

    def test_npu_acquire_downloads_verifies_and_reuses_complete_fixture(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2-npu')
        model = self.catalog.models[profile['model']]
        payloads = {}
        for entry in model['files']:
            suffix = Path(entry['path']).suffix
            payload = (b'HGN1' + struct.pack('<IQ', 2, 3) + b'fixture' if suffix == '.hgn'
                       else b'{}' if suffix == '.json'
                       else b'messages enable_thinking' if suffix == '.jinja' else b'fixture')
            entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            url = artifact_store.Artifact.from_entry(self.root / entry['path'], entry,
                                                    model['download']['repository'], model['download']['revision']).source
            payloads[url] = payload
        manifest = copy.deepcopy(cli.halogen_npu.MANIFEST)
        for value in manifest['models'].values():
            for entry in value['files']:
                payload = b'npu fixture'
                entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                payloads[f"https://huggingface.co/{value['repo']}/resolve/{value['revision']}/{entry['path']}"] = payload

        def response(request, timeout):
            result = io.BytesIO(payloads[request.full_url])
            result.status = 200
            result.getcode = lambda: 200
            return result

        with mock.patch.object(cli.halogen_npu, 'MANIFEST', manifest), self.installed_images(), \
                mock.patch.object(cli, 'ensure_model_mount'), \
                mock.patch.object(cli, 'command_runtime_install', return_value=0) as install, \
                mock.patch.object(artifact_store.urllib.request, 'urlopen', side_effect=response) as network, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.command_profile_acquire(self.config, self.catalog, profile, dry_run=False), 0)
            closure = cli.resolve_artifacts(self.config, self.catalog, profile)
            expected = [a for selection in closure.models for a in selection.downloads()] + list(closure.auxiliary)
            self.assertEqual(network.call_count, len(expected))
            self.assertEqual(artifact_store.verify(tuple(expected), full=True), [])
            self.assertEqual(cli.profile_acquisition_plan(self.config, self.catalog, profile)['total_additional_download_bytes'], 0)
            network.reset_mock()
            self.assertEqual(cli.command_profile_acquire(self.config, self.catalog, profile, dry_run=False), 0)
            network.assert_not_called()
            self.assertEqual(install.call_count, 2)
            # A damaged auxiliary file must be repaired before runtime installation.
            closure.auxiliary[0].destination.write_bytes(b'damaged')
            network.side_effect = None
            corrupt = io.BytesIO(b'bad replacement')
            corrupt.status = 200
            corrupt.getcode = lambda: 200
            network.return_value = corrupt
            with self.assertRaises(cli.HaloError):
                cli.command_profile_acquire(self.config, self.catalog, profile, dry_run=False)
            self.assertEqual(install.call_count, 2)

    def test_readiness_requires_every_auxiliary_artifact(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2-npu')
        with self.installed_images(), mock.patch.object(cli, 'verify_model', return_value={'valid': True}), \
                mock.patch.object(cli, 'meminfo_bytes', return_value=128 * 1024**3), \
                mock.patch.object(cli.halogen_npu, 'host_errors', return_value=[]):
            ready, reason = cli.profile_availability(self.config, self.catalog, profile)
        self.assertFalse(ready)
        for artifact in cli.resolve_artifacts(self.config, self.catalog, profile).auxiliary:
            self.assertIn(str(artifact.destination), reason)

    def test_npu_only_downloader_rejects_corrupt_download(self):
        root = self.root / 'npu'
        with mock.patch.object(artifact_store.urllib.request, 'urlopen') as network, \
                contextlib.redirect_stdout(io.StringIO()):
            response = io.BytesIO(b'corrupt')
            response.status = 200
            response.getcode = lambda: 200
            network.return_value = response
            with self.assertRaises(cli.HaloError):
                cli.halogen_npu.download(root, ['qwen3-reranker-0.6b'])
        first = cli.halogen_npu.pinned_artifacts(root, ['qwen3-reranker-0.6b'])[0]
        self.assertFalse(first.destination.exists())
        self.assertFalse(first.destination.with_name('.' + first.destination.name + '.partial').exists())

    def test_npu_root_cannot_escape_via_symlink(self):
        outside = Path(self.tmp.name) / 'outside'
        outside.mkdir()
        (self.root / 'halogen-npu').symlink_to(outside, target_is_directory=True)
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2-npu')
        with self.assertRaises(cli.HaloError):
            cli.resolve_artifacts(self.config, self.catalog, profile)

    def test_registered_ports_are_validated_before_dispatch(self):
        values = dict(self.config.values)
        values['HALOGEN_API_PORT'] = '70000'
        with self.assertRaises(cli.HaloError):
            cli.validate_config(cli.Config(values, self.config.files))

    def test_runtime_and_auxiliary_manifest_releases_match(self):
        self.assertEqual(engine_halogen.RELEASE, cli.halogen_npu.MANIFEST['release'])
        self.assertEqual(engine_halogen.IMAGE_DIGEST, cli.halogen_npu.MANIFEST['image_digest'])

    def test_runtime_upgrade_reuses_existing_npu_artifact_directory(self):
        profile = cli.resolve_profile(self.catalog, 'qwen3.8-halogen-v2-npu')
        with mock.patch.dict(cli.halogen_npu.MANIFEST, {'release': 'future-runtime'}):
            closure = cli.resolve_artifacts(self.config, self.catalog, profile)
        self.assertEqual(closure.auxiliary_root, self.root / 'halogen-npu' / '0.17.4')
        self.assertTrue(all(a.destination.is_relative_to(closure.auxiliary_root) for a in closure.auxiliary))
