import os
from setuptools import setup, find_packages


with open('VERSION', 'r') as f:
    version = f.read().rstrip()

install_requires = [
    'numpy',
    'pandas',
    'scikit-learn',
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
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

try:
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    CHANGES = ''


setup(name='dstoolbox',
      version=version,
      description='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': tests_require,
          'docs': docs_require,
          },
      entry_points={
          },
      )
