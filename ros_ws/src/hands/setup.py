import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hands'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        
        # Install rviz config files
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),

        # Install Inspire Hand files
        (os.path.join('share', package_name, 'urdf/inspire'), glob('urdf/inspire/*.urdf')),
        (os.path.join('share', package_name, 'urdf/inspire/meshes/visual'), glob('urdf/inspire/meshes/visual/*')),
        (os.path.join('share', package_name, 'urdf/inspire/meshes/collision'), glob('urdf/inspire/meshes/collision/*')),

        # Install Leap Hand files
        (os.path.join('share', package_name, 'urdf/leap'), glob('urdf/leap/*.urdf')),
        (os.path.join('share', package_name, 'urdf/leap/meshes'), glob('urdf/leap/meshes/*.stl')),
        (os.path.join('share', package_name, 'urdf/leap/meshes/visual'), glob('urdf/leap/meshes/visual/*')),
        (os.path.join('share', package_name, 'urdf/leap/meshes/collision'), glob('urdf/leap/meshes/collision/*')),

        # Install ORCA Hand files
        (os.path.join('share', package_name, 'urdf/orca'), glob('urdf/orca/*.urdf')),
        (os.path.join('share', package_name, 'urdf/orca/visual'), glob('urdf/orca/visual/*.stl')),
        (os.path.join('share', package_name, 'urdf/orca/collision'), glob('urdf/orca/collision/*.stl')),

        # Install Nano Hand files
        (os.path.join('share', package_name, 'urdf/nano'), glob('urdf/nano/*.urdf')),
        (os.path.join('share', package_name, 'urdf/nano/meshes'), glob('urdf/nano/meshes/*.stl')),

        # Install CSV files
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armlab',
    maintainer_email='armlab@todo.todo',
    description='URDF descriptions and controllers for various robot hands.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_controller = hands.hand_controller:main',
            'hand_controller_ik = hands.hand_controller_ik:main',
            'rokoko_listener = hands.rokoko_listener:main',
            'visualization_node = hands.visualization_node:main',
            'orca_retargeting = hands.orca_retargeting:main',
            'inspire_retargeting = hands.inspire_retargeting:main',
            'leap_retargeting = hands.leap_retargeting:main',
            'nano_retargeting = hands.nano_retargeting:main',
            'nano_retargeting_physics = hands.nano_retargeting_physics:main',
            'control_type_publisher = hands.control_type_publisher:main',
            'orca_hardware_control = hands.orca_hardware_control:main',
            'fingertip_error_plotter = hands.fingertip_error_plotter:main',
            'inspire_hardware_control = hands.inspire_hardware_control:main',
            'leap_hardware_control = hands.leap_hardware_control:main',
            'vive_listener = hands.vive_listener:main',
            'nano_hardware_control = hands.nano_hardware_control:main'
        ],
    },
)

