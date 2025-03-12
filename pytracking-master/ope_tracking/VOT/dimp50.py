import os
import sys 

oracle_libs = os.environ['ORACLE_HOME']+"/lib/"
rerun = True

if not 'LD_LIBRARY_PATH' in os.environ:
  os.environ['LD_LIBRARY_PATH'] =":"+oracle_libs
elif not oracle_libs in os.environ.get('LD_LIBRARY_PATH'):
  os.environ['LD_LIBRARY_PATH'] +=":"+oracle_libs
else:
  rerun = False

if rerun:
  os.execve(os.path.realpath(__file__), sys.argv, os.environ)

print(os.environ['LD_LIBRARY_PATH'])


sys.path.append('/home/zhanghc@1/888_second_work/0.second/pytracking-master')
import pytracking.run_vot as run_vot
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

run_vot.run_vot2021('dimp', 'dimp50')