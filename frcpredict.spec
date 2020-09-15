# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

datas = [("data", "data")]
for ui_file_path in Path("frcpredict").rglob("*.ui"):
    datas.append((ui_file_path, ui_file_path.parents[0]))

block_cipher = None

a = Analysis(["frcpredict/__main__.py"],
             binaries=[],
             datas=datas,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=["matplotlib"],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.binaries = TOC([x for x in a.binaries if (
    not x[0].lower().startswith("api-ms-") and
    not x[0].lower().startswith("d3dcompiler") and
    not x[0].lower().startswith("msvcp140") and
    not x[0].lower().startswith("ucrtbase") and
    not x[0].lower().startswith("vcomp140") and
    not x[0].lower().startswith("vcruntime140")
)])

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name="ResolutionPredictionSoftware",
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)

if sys.platform != "darwin":
    coll = COLLECT(exe,
                   a.binaries,
                   a.zipfiles,
                   a.datas,
                   strip=False,
                   upx=True,
                   upx_exclude=[],
                   name="ResolutionPredictionSoftware")
else:
    app = BUNDLE(exe,
                 a.binaries,
                 a.zipfiles,
                 a.datas,
                 name="Resolution Prediction Software.app",
                 icon=None,
                 bundle_identifier=None,
                 info_plist={
                    "NSPrincipalClass": "NSApplication",
                    "NSHighResolutionCapable": "True"
                 })
