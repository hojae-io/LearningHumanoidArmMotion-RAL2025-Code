import os
from extensions import ISAACLAB_BRL_ROOT_DIR
import shutil
import fnmatch

# create ignore patterns dynamically based on include patterns
def create_ignored_pattern_except(*patterns):
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns for name in
                    fnmatch.filter(names, pattern))
        ignore = set(name for name in names if name not in keep and
                        not os.path.isdir(os.path.join(path, name)))
        return ignore
    return _ignore_patterns

def remove_empty_folders(path, removeRoot=True):
    if not os.path.isdir(path):
        return
    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)
    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        os.rmdir(path)

def log_and_save(log_dir):
    """Configure local code logging"""

    extensions_dir = os.path.join(ISAACLAB_BRL_ROOT_DIR, 'extensions')
    extensions_target = os.path.join('extensions')

    rsl_rl_dir = os.path.join(ISAACLAB_BRL_ROOT_DIR, 'rsl_rl')
    rsl_rl_target = os.path.join('rsl_rl')

    # list of things to copy
    # source paths need the full path and target are relative to log_dir
    save_paths = [
        {'type': 'dir', 'source_dir': extensions_dir,
                        'target_dir': extensions_target,
            'include_patterns': ('*.py', '*.json')},
        {'type': 'dir', 'source_dir': rsl_rl_dir,
                        'target_dir': rsl_rl_target,
            'include_patterns': ('*.py', '*.json')}
    ]

    # copy the relevant source files to the local logs for records
    save_dir = log_dir+'/files/'
    for save_path in save_paths:
        if save_path['type'] == 'file':
            os.makedirs(save_dir+save_path['target_dir'],
                        exist_ok=True)
            shutil.copy2(save_path['source_file'],
                            save_dir+save_path['target_dir'])
        elif save_path['type'] == 'dir':
            shutil.copytree(
                save_path['source_dir'],
                save_dir+save_path['target_dir'],
                ignore=create_ignored_pattern_except(
                    *save_path['include_patterns']))
        else:
            print('WARNING: uncaught save path type:', save_path['type'])
            
    remove_empty_folders(save_dir)

    print('[INFO] Saved code to:', save_dir)