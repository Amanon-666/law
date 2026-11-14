"""Bounded-run phase diagnostics; identical CUDA prediction computation."""
from contextlib import ExitStack
import faulthandler
import json
import os
import joblib
from pathlib import Path
import time
from unittest.mock import patch

import fields
import history_fit
import frozen_hurdle_translator as frozen
from paper_bench_20261111 import ext_gp_cuda


def log(stage, **details):
    if os.environ.get('REFI_GP_TRACE', '1') == '0':
        return
    status = Path('/proc/self/status').read_text().splitlines()
    memory = {line.split(':')[0]: line.split(':', 1)[1].strip()
              for line in status if line.startswith(('VmRSS:', 'VmHWM:'))}
    print(json.dumps(dict(stage=stage, time=time.time(), **memory, **details)), flush=True)


def trace(name, fn):
    def call(*args, **kwargs):
        detail = {}
        if name == 'History.init':
            detail['positions'] = len(args[1])
        if name == 'infer.cuda':
            detail['positions'] = len(args[0])
        log(name + '.begin', **detail)
        cache = None
        if name in ('refi.prepare', 'refi.predict'):
            # Includes model/Prepared object, all H/S or Q arrays, and arguments.
            key = joblib.hash((args, kwargs))
            cache = Path(os.environ['REFI_DIAGNOSTIC_CACHE']) / (name + '-' + key + '.joblib')
            if cache.exists():
                result = joblib.load(cache)
                log(name + '.cache_hit', path=str(cache))
                return result
        result = fn(*args, **kwargs)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, cache)
            log(name + '.cached', path=str(cache))
        log(name + '.end', **detail)
        return result
    return call


class ReFiGPCUDADiagnostic(ext_gp_cuda.ReFiGPCUDA):
    def predict(self, task):
        if os.environ.get('REFI_GP_TRACE', '1') != '0':
            faulthandler.enable()
            faulthandler.dump_traceback_later(60, repeat=True)
        log('task.begin', task_id=task.task_id)
        try:
            with ExitStack() as stack:
                for owner, attr, name in (
                    (frozen.FrozenHurdleTranslator, 'prepare', 'refi.prepare'),
                    (frozen.PreparedFrozenHurdle, 'predict', 'refi.predict'),
                    (fields.History, '__init__', 'History.init'),
                    (history_fit, 'single_history', 'single_history'),
                    (ext_gp_cuda, 'infer', 'infer.cuda'),
                ):
                    stack.enter_context(patch.object(owner, attr, trace(name, getattr(owner, attr))))
                return super().predict(task)
        finally:
            faulthandler.cancel_dump_traceback_later()
            log('task.end', task_id=task.task_id)
