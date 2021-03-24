"""
Trading-Technical-Indicators (tti) python library

tti is a python library for calculating more than 60 trading technical
indicators from stocks data. The library provides an API for:

* trading technical indicators value calculation
* trading technical indicators graph preparation
* trading signal calculation
* trading simulation based on trading signals
* prices direction prediction based on machine learning algorithms

Project site is https://www.trading-technical-indicators.org/
"""

from tti import indicators
from tti import utils
from tti import ml

__version__ = '1.0.0'

__all__ = ['indicators', 'utils', 'ml']
