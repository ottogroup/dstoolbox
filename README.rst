Otto Group BI Data Science Toolbox
==================================

NOTE: THIS IS NOT YET RELEASE READY, PLEASE BE PATIENT.

This repository contains tools that make working with
`scikit-learn <http://scikit-learn.org/>`__ and
`pandas <http://pandas.pydata.org/>`__ easier.

|Build Status|

What is this?
-------------

dstoolbox is not one big tool but rather an amalgamation of small
re-usable tools. They are intended to work well with scikit-learn and
pandas make the integration of those libraries easier.

The best way to get started is to have a look at the `notebooks
folder <https://github.com/ottogroup/dstoolbox/tree/master/notebooks>`__,
especially at the `showcase
notebook <https://github.com/ottogroup/dstoolbox/blob/master/notebooks/Showcase.ipynb>`__.

The tools included here are used by us at Otto Group BI for our
production services, as well as by individual members for machine
learning related things, such as participating in Kaggle competitions.

Installation instructions
-------------------------

Using ``pip``::

  pip install dstoolbox


There is a conda recipe for those who want to build their own conda
package.


Contributing
------------

Pull requests are welcome. Here are some directions:

Tests
~~~~~

To run the tests, you need to install the dev requirements using pip::

  pip install -r requirements-dev.txt

or conda::

  conda install --file requirements-dev.txt

Next you should check that all unit tests and all static code checks
pass::

  py.test
  pylint dstoolbox

Guidelines
~~~~~~~~~~

-  Python 3 only.

-  Code should be re-usable and succinct.

-  Where applicable, it should be compatible with
   `scikit-learn <http://scikit-learn.org/>`__,
   `pandas <http://pandas.pydata.org/>`__, and
   `Palladium <https://github.com/ottogroup/palladium>`__.

-  It should be documented and unit-tested using pytest (100% code
   coverage desired).

-  It should conform to the coding standards prescribed by pylint (where
   it makes sense).

-  There should be usage examples that cover the most common use cases
   (the best place would be an IPython/Jupyter notebook).

-  Don't add dependencies unless absolutely necessary.


.. |Build Status| image:: https://travis-ci.org/ottogroup/dstoolbox.svg?branch=master
   :target: https://travis-ci.org/ottogroup/dstoolbox
