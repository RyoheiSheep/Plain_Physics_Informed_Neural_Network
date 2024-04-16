import re
from setuptools import setup,find_packages
import sys

TORCH_AVAILABLE=True

try:
    import torch
    from torch.utils import cpp_extension
except ImportError:
    TORCH_AVAILABLE=False
    print("[WARNING] Unable to import torch, pre-compiling ops will be disabled.")


def get_package_dir():
    pkg_dir={
            "pinn.tools":"tools",
            "pinn.exp.default":"exps/default",
            }
    return pkg_dir

def get_install_requirements():
    with open("requirements.txt","r",encoding="utf-8") as f:
        reqs=[x.strip() for x in f.read().splitlines()]

    reqs=[x for x in reqs if not x.startswith("#")]
    return reqs

def get_pinn_version():
    with open("pinn/__init__.py","r") as f:
        version=re.search(
                r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                f.read(),re.MULTILINE
                ).group(1)
        return version

def get_long_description():
    with open("README.md","r",encoding="utf-8") as f:
        long_description=f.read()
    return long_description

# def get_ext_modules():
#     ext_module=[]
#     if sys.platform !="

setup(
        name="pinn",
        version=get_pinn_version(),
        author="Ryohei Yamada",
        desription="Plain PINN package for personal use",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
       )

