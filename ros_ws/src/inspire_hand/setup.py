from setuptools import setup

package_name = 'inspire_hand'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'rokoko_csv'), glob('rokoko_csv/*.csv')),
        (os.path.join('share', package_name, 'rokoko_csv/legacy'), glob('rokoko_csv/legacy/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='belle',
    maintainer_email='yanisa@stanford.com',
    description='CSV → Inspire Hand controller',
    license='MIT',
    entry_points={
        'console_scripts': [
            "inspire_hand_listener = inspire_hand.inspire_hand_listener:main",
            "rokoko_to_inspire = inspire_hand.rokoko_to_inspire:main",
            "EMG_to_inspire = inspire_hand.EMG_to_inspire:main",
        ],
    },
)
