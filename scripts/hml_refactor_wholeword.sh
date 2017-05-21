#!/bin/bash

if [ $# -ne 2 ] ; then
  echo "The first parameter is the old string to be replaced"
  echo "The second parameter is the new string replacing the old one"
  echo "Example: $0 old_string new_string"
  exit 0
fi

cd ../src && grep -sl $1 *.cu *.c *.cpp *.h *.hpp | xargs sed -i "s/\b$1\b/$2/g"

cd ../include && grep -sl $1 *.cu *.c *.cpp *.h *.hpp | xargs sed -i "s/\b$1\b/$2/g"

#eof;;
