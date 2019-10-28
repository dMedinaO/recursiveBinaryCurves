#/usr/bin/sh

pip uninstall RECURSIVE-CLUSTERING-MODULES
rm /usr/local/lib/python2.7/dist-packages/modulesRV -R
python setup.py install
