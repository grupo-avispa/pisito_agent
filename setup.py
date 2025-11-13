from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'pisito_agent'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('lib/' + package_name, [package_name + '/langgraph_home_assistant.py']),
        ('share/' + package_name, ['.env']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'templates'),
            glob(os.path.join('templates', '*.jinja'))),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.json'))),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.yaml')))
    ],
    install_requires=[
        'setuptools',
        'jinja2',
    ],
    zip_safe=False,
    maintainer='Oscar Pons Fernandez',
    maintainer_email='opfernandez@uma.es',
    description='ROS2 smolagents agent',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'langgraph_ros_home_assistant_agent = ' + package_name + '.langgraph_ros_home_assistant_agent:main',
        ],
    },
)
