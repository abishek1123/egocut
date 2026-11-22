"""Runs our implementation of the kit's output checks (code/validate_local.py).
The organisers did not ship validate.py, so this makes the documented command work as written:
    python validate.py --dir outputs/
"""
import os, runpy, sys
sys.argv[0] = 'validate.py'
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code', 'validate_local.py'), run_name='__main__')
