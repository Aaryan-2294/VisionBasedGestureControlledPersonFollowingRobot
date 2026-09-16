from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vbgc_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'description'),
            glob('description/*')
        ),
        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aaryan Patel',
    maintainer_email='aaryanpatel2294@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'gesture_publisher = vbgc_robot.gesture_publisher:main',
        'camera_target_publisher = vbgc_robot.camera_target_publisher:main',
        'simulated_target = vbgc_robot.simulated_target:main',
        'following_controller = vbgc_robot.following_controller:main',
        ],
    },
)