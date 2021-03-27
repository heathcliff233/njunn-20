#!/usr/bin/env bash
pip install nbdime torch opencv-python matplotlib numpy tensorflow-gpu==1.15.0 runipy
runipy ./njunn-as2.ipynb --html report.html
