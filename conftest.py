# Adds src/ to sys.path so tests can import the modules directly.
# Required because the brief mandates a src/ subdirectory but does not
# install the project as a package.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
