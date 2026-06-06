# Builds a single self-contained Singularity desktop app (no Python needed by
# the end user). Build ON THE SAME OS as the target machine.
from PyInstaller.utils.hooks import collect_submodules

hidden = (collect_submodules('ib_async') + collect_submodules('uvicorn')
          + collect_submodules('webview') + [
    'eventkit', 'uvicorn.logging', 'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
])

a = Analysis(['run.py'], pathex=['.'], binaries=[],
             datas=[('frontend', 'frontend')],
             hiddenimports=hidden, hookspath=[], runtime_hooks=[], excludes=[])
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
          name='Singularity', debug=False, strip=False, upx=True,
          console=False)   # console=False -> clean window, no terminal box
