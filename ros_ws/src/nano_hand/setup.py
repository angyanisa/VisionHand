from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nano_hand'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rokoko_csv'), glob('rokoko_csv/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='belle',
    maintainer_email='yanisa@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            "nano_hand_listener = nano_hand.nano_hand_listener:main",
            "emg_to_nano = nano_hand.EMG_to_nano:main",
        ],
    },
)
