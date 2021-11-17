#  MultiColor Exoplanet Validation tool
#  Copyright (C) 2021  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup

version = {}
with open("mcv/version.py") as fp:
    exec(fp.read(), version)

setup(name='MCV',
      version=str(version['__version__']),
      description='Multicolor transiting exoplanet candidate validation tool.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/mcv',
      package_dir={'mcv':'mcv'},
      packages=['mcv', 'mcv.io'],
      install_requires=["semantic_version"],
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
      ],
      keywords='astronomy astrophysics exoplanets'
      )
