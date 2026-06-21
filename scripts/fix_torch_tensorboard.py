"""Fix PyTorch 1.9.1 tensorboard compatibility issue.

PyTorch 1.9.1 uses `from setuptools import distutils` then accesses
`distutils.version.LooseVersion`, which fails because the distutils
submodule `version` is not automatically exposed as an attribute.
This script patches the import to use `from distutils.version import LooseVersion`.
"""
import pathlib
import torch

target = pathlib.Path(torch.__path__[0]) / "utils" / "tensorboard" / "__init__.py"
text = target.read_text()

old = "from setuptools import distutils\n\nLooseVersion = distutils.version.LooseVersion"
new = "from distutils.version import LooseVersion"

if old in text:
    text = text.replace(old, new).replace("del distutils\n", "")
    target.write_text(text)
    print(f"Patched: {target}")
elif new in text:
    print(f"Already patched: {target}")
else:
    print(f"Warning: unexpected content in {target}, skipping.")
