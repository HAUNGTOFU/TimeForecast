import setuptools
with open("README", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='Timeforecast',
    version='0.1',
    description='model for time forecast',
    author='huangtofu',
    author_email='2672916471@qq.com',
    url='https://github.com/HAUNGTOFU/TimeForecast',
    zip_safe=False,
    packages=setuptools.find_packages())
