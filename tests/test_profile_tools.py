import sqlite3
from contextlib import closing
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from profile_inference import native_command,ncu_command,sqlite_activity
from benchmark_worker import profile_control

class ProfileToolsTests(unittest.TestCase):
    def command(self,tokens=512,optimized=True):
        return native_command(Path('/artifact'),Path('/model'),Path('/request.json'),Path('/run'),tokens,20480,optimized,'/tool/nsys')
    def test_capture_is_separate_from_synchronizing_diagnostics(self):
        cmd=self.command()
        self.assertIn('--nsys-profile-dir',cmd)
        self.assertEqual(cmd[cmd.index('--warmup')+1],'1')
        for flag in ('--profile-kernels','--profile-stages','--cache'):
            self.assertNotIn(flag,cmd)
        self.assertIn('--mtp-verify-graph',cmd)
    def test_first_token_has_no_unused_verify_graph(self):
        cmd=self.command(1)
        self.assertNotIn('--mtp-verify-graph',cmd)
        self.assertEqual(cmd[cmd.index('--max-new-tokens')+1],'1')
    def test_reference_and_optimized_differ_only_by_new_flags(self):
        ref=self.command(512,False);opt=self.command()
        self.assertEqual(opt[:-2],ref)
        self.assertEqual(opt[-2:],['--fused-residual-norm','--gpu-candidates'])
    def test_ncu_profiles_isolated_kernels_not_mpi(self):
        for name in ('fp8','gdn-solve','gdn-state'):
            cmd=ncu_command('/tool/ncu',name,Path('/run'))
            self.assertNotIn('mpirun',cmd);self.assertNotIn('build/avi-worker',cmd)
            self.assertIn('--launch-count',cmd);self.assertIn('--kernel-name',cmd)
            self.assertEqual(cmd[cmd.index('--launch-count')+1],'2')
    def test_empty_capture_does_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.sqlite'
            self.assertEqual(sqlite_activity(path)['status'],'missing_sqlite')
            with closing(sqlite3.connect(path)) as db, db:db.execute('CREATE TABLE other(x)')
            self.assertEqual(sqlite_activity(path)['status'],'missing_kernel_table')
            with closing(sqlite3.connect(path)) as db, db:db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER)')
            self.assertEqual(sqlite_activity(path)['status'],'empty_capture')
    def test_real_kernel_activity_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.sqlite'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER)')
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)',[(10,30),(20,50)])
            record=sqlite_activity(path)
            self.assertEqual(record['kernel_count'],2)
            self.assertEqual(record['status'],'captured')
            self.assertAlmostEqual(record['kernel_span_seconds'],40/1e9)

    def test_profile_protocol_waits_for_acknowledgement(self):
        for start in (True,False):
            sent=[]
            events=iter([{'event':'stats'},{'event':'profile_started' if start else 'profile_stopped'}])
            profile_control(sent.append,lambda deadline:next(events),start,1)
            self.assertEqual(sent,[{'op':'profile_start' if start else 'profile_stop'}])
    def test_profile_protocol_propagates_worker_failure(self):
        def fail(deadline):raise RuntimeError('Worker exited')
        with self.assertRaises(RuntimeError):profile_control(lambda x:None,fail,True,1)
