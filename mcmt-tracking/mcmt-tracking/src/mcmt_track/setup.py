import os
from setuptools import setup
from glob import glob

package_name = 'mcmt_track'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='niven',
    maintainer_email='sieniven@gmail.com',
    description='Re-identification package for multi-camera multi-target tracking',
    license='Temasek Laboratories, Centre of Flight Sciences',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mcmt_single_tracker = mcmt_track.mcmt_single_tracker_main:main',
            'mcmt_multi_tracker = mcmt_track.mcmt_multi_tracker_main:main',
        ],
    },
)
