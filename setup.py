from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'drims_homework'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'trees'), glob('trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kalman',
    maintainer_email='samuele.sandrini@stiima.cnr.it',
    description='Drims2 Homework Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'demo_node = drims_homework.demo_node:main',
            'dice_manipulation_node = drims_homework.dice_manipulation_node:main',
            'dice_task_orchestrator = drims_homework.dice_task_orchestrator:main',
            'dice_simulator_healer = drims_homework.dice_simulator_healer:main',
            'yellow_dice_localizer = drims_homework.yellow_dice_localizer:main',
        ],
    },
)
