#!/usr/bin/env python

import datetime
import os
import signal
import subprocess
import re
import time


eps = 1e-5


def save_arg_dict(d, base_dir='./', filename='args.txt', log=True):
    def _format_value(vx):
        if isinstance(v, float):
            return '%.8f' % vx
        elif isinstance(v, int):
            return '%d' % vx
        else:
            return '%s' % str(vx)

    with open(os.path.join(base_dir, filename), 'w') as f:
        for k, v in d.items():
            f.write('%s\t%s\n' % (k, _format_value(v)))
    if log:
        print('Saved settings to %s' % os.path.join(base_dir, filename))


def mkdir_p(path, log=True):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    if log:
        print('Created directory %s' % path)


def date_filename(base_dir='./', prefix=''):
    dt = datetime.datetime.now()
    return os.path.join(base_dir, '{}{}_{:02d}-{:02d}-{:02d}'.format(
        prefix, dt.date(), dt.hour, dt.minute, dt.second))


def setup(args):
    """Boilerplate setup, returning dict of configured items."""
    # log directory
    log_directory = date_filename(args.base_log_dir, args.expr_name)
    mkdir_p(log_directory)
    save_arg_dict(args.__dict__, base_dir=log_directory)
    # kid directory
    return dict(log_directory=log_directory, kid_directory=args.base_kid_dir)


REGEX_TH = re.compile(r'throughput\(([^)]+)\)')
REGEX_ABORT = re.compile(r'agg_abort_rate\(([^)]+)\)')


def parse(return_string):
    if return_string is None:
        return 0.0, 0.0
    parse_th = re.search(REGEX_TH, return_string)
    parse_abort = re.search(REGEX_ABORT, return_string)
    if parse_th is None or parse_abort is None:
        return float(.0), float(.0)
    th = parse_th.groups()[0]
    abort = parse_abort.groups()[0]
    return float(th), float(abort)


def run(command, die_after=0):
    extra = {} if die_after == 0 else {'preexec_fn': os.setsid}
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True, **extra)
    for _ in range(die_after if die_after > 0 else 600):
        if process.poll() is not None:
            break
        time.sleep(1)
    out_code = -1 if process.poll() is None else process.returncode
    if out_code < 0:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            print('{}, but continuing'.format(e))
        assert die_after != 0, 'Should only time out with die_after set'
        print('Failed with return code {}'.format(process.returncode))
        print("running = ", command)
        process.stdout.flush()
        return process.stdout.read().decode('utf-8')
    elif out_code > 0:
        print('Failed with return code {}'.format(process.returncode))
        print("running = ", command)
        process.stdout.flush()
        return process.stdout.read().decode('utf-8')
    process.stdout.flush()
    return process.stdout.read().decode('utf-8')
