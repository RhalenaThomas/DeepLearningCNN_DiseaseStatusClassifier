#!/bin/bash

original_path=$PWD
mkdir .local
cd .local

mkdir ssl
mkdir python
mkdir src
cd src 


wget https://www.python.org/ftp/python/3.9.13/Python-3.9.13.tgz
wget https://www.openssl.org/source/openssl-1.1.1d.tar.gz

tar zxfv Python-3.9.13.tgz
tar xvfz openssl-1.1.1d.tar.gz

cd openssl-1.1.1d
./config --prefix=$original_path/.local/ssl
make && make install

cd $original_path/.local/src/

find ./Python-3.9.13/Python -type d | xargs chmod 0755
cd Python-3.9.13
./configure --prefix=$original_path/.local/python --with-openssl=$original_path/.local/ssl  --enable-optimizations
make && make install

export PATH=$CWD:$PATH  
