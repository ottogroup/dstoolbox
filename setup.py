import os
from setuptools import setup, find_packages


with open('VERSION', 'r') as f:
    version = f.read().rstrip()

install_requires = [
    'numpy',
    'pandas',
    'scikit-learn>=0.21,<0.23dev0',
    'scipy',
    ]

tests_require = [
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''

try:
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    CHANGES = ''


setup(
    name='dstoolbox',
    version=version,
    description=(
        'Tools that make working with scikit-learn and pandas easier.'),
    author='Otto Group',
    author_email='benjamin.bossan@ottogroup.com',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/ottogroup/dstoolbox',
    download_url='https://github.com/ottogroup/dstoolbox/tarball/' + version,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
        },
    setup_requires=["setuptools_git >= 0.3"],
    entry_points={},
    )
