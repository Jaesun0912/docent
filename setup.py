#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from distutils.core import setup

setup(
    name="docent",
    version='0.2.0',
    packages=[
        'docent',
        'docent.monte_carlo',
        'docent.structure',
        'docent.scripts',
        'docent.util',
    ],
    entry_points={
        "console_scripts": [
            "docent = docent.scripts.main:main"
        ]
    },
)

