#!/usr/bin/env python3
"""Allow one rootless operator to lock Halogen NPU buffers, now and at login."""
import argparse
import datetime
import os
from pathlib import Path
import pwd
import resource
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', required=True)
    parser.add_argument('--session-pid', type=int, help='Also raise this existing operator process limit; children inherit it')
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('run through pkexec or sudo')
    account = pwd.getpwnam(args.user)
    if account.pw_uid == 0 or any(c.isspace() for c in args.user):
        parser.error('select a non-root operator')
    manager = int(subprocess.check_output(['systemctl', 'show', f'user@{account.pw_uid}.service', '-p', 'MainPID', '--value']))
    processes = [p for p in [manager, args.session_pid] if p]
    for pid in processes:
        if Path(f'/proc/{pid}').stat().st_uid != account.pw_uid:
            parser.error(f'process {pid} does not belong to {args.user}')
    files = [
        (Path('/etc/security/limits.d/90-halo-ai-memlock.conf'), f'{args.user} soft memlock unlimited\n{args.user} hard memlock unlimited\n', 0, 0),
        (Path(f'/etc/systemd/system/user@{account.pw_uid}.service.d/90-halo-ai-memlock.conf'), '[Service]\nLimitMEMLOCK=infinity\n', 0, 0),
        (Path(account.pw_dir) / '.config/systemd/user.conf.d/90-halo-ai-memlock.conf', '[Manager]\nDefaultLimitMEMLOCK=infinity\n', account.pw_uid, account.pw_gid),
    ]
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for path, content, uid, gid in files:
        if path.is_symlink():
            parser.error(f'refusing symlink: {path}')
        missing = []
        parent = path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        for directory in missing:
            os.chown(directory, uid, gid)
        if path.exists() and path.read_text() != content:
            shutil.copy2(path, path.with_name(path.name + '.backup-' + timestamp))
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.halo-memlock-')
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(content)
        os.chmod(temporary, 0o644)
        os.chown(temporary, uid, gid)
        os.replace(temporary, path)
        print(f'Configured {path}', flush=True)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    for pid in processes:
        resource.prlimit(pid, resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        print(f'Raised memlock for operator process {pid}', flush=True)
    if manager:
        subprocess.run(['runuser', '-u', args.user, '--', 'env', f'XDG_RUNTIME_DIR=/run/user/{account.pw_uid}', 'systemctl', '--user', 'daemon-reexec'], check=True)
    print('Future logins and user services have unlimited memlock. Existing processes need a new login or an explicit --session-pid adjustment.')


if __name__ == '__main__':
    main()
