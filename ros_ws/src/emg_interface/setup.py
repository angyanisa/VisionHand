from setuptools import setup

package_name = 'emg_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    package_data={
        package_name: ['*.pkl'],  # Include all .pkl files in the package
    },
    maintainer='belle',
    maintainer_email='yanisa@stanford.com',
    description='EMG interface',
    license='MIT',
    entry_points={
        'console_scripts': [
            "emg_udp_collector = emg_interface.EMG_udp_collector:main",
            "emg_feature_extractor = emg_interface.EMG_feature_extractor:main",
            "emg_train_model = emg_interface.EMG_train_model:main",
            "emg_live_classifier = emg_interface.EMG_live_classifier:main",
        ],
    },
)
