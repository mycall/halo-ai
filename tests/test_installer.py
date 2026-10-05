import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallerCatalogTests(unittest.TestCase):
    def setUp(self):
        source = (ROOT / 'install.sh').read_text()
        self.functions = '\n'.join(re.search(r'^' + name + r'\(\) \{\n.*?^\}', source, re.M | re.S).group() for name in ('catalog_destination_name', 'verify_catalogs', 'install_catalogs'))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'models.d').mkdir()

    def run_shell(self, body):
        return subprocess.run(['bash', '-c', self.functions + '\n' + body], env={**os.environ, 'source_root': str(ROOT), 'CONFIG_ROOT': str(self.root)}, capture_output=True, text=True)

    def test_installs_every_catalog_preserving_legacy_filename(self):
        result = self.run_shell('run_action() { shift; install -m 0644 "${@: -2}"; }; install_catalogs; verify_catalogs')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'models.d/10-halogen.json').read_bytes(), (ROOT / 'config/models.d/halogen.json').read_bytes())
        self.assertTrue((self.root / 'models.d/00-strix-halo.json').exists())

    def test_verification_rejects_missing_or_stale_supplemental_catalog(self):
        (self.root / 'models.d/00-strix-halo.json').write_bytes((ROOT / 'config/models.d/strix-halo.json').read_bytes())
        self.assertNotEqual(self.run_shell('verify_catalogs').returncode, 0)
        (self.root / 'models.d/10-halogen.json').write_text('{}')
        self.assertNotEqual(self.run_shell('verify_catalogs').returncode, 0)


class HalogenMigrationTests(unittest.TestCase):
    def test_old_pin_migrates_and_custom_pin_is_preserved(self):
        source = (ROOT / 'install.sh').read_text()
        function = re.search(r'^migrate_halogen_image\(\) \{\n.*?^\}', source, re.M | re.S).group()
        expected = re.search(r'^HALOGEN_IMAGE=(.*)$', (ROOT / 'config/halo-ai.env.example').read_text(), re.M).group(1)
        for original in ['', 'ghcr.io/peonist-ai/halogen-flash-server:0.12.2', 'custom-image:local']:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / 'config.env'
                config.write_text('KEEP=this\n' + (f'HALOGEN_IMAGE={original}\n' if original else ''))
                result = subprocess.run(['bash', '-c', function + '\nchown() { :; }; dry_run=false; migrate_halogen_image'], env={**os.environ, 'source_root': str(ROOT), 'CONFIG_ROOT': temporary, 'run_gid': str(os.getgid())}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('KEEP=this\n', config.read_text())
                self.assertIn('HALOGEN_IMAGE=' + (original if original.startswith('custom') else expected), config.read_text())
