#!/bin/bash

if [ $# -ne 0 ] ; then
  echo "This script takes no argument and formats all .h, .c, and .cu files "
  echo "under /include and /src using Astyle."
  echo "Example: $0"
  exit 0
fi

for dir in src include
do
  echo "formatting files in" $dir
  cd ../$dir && ls -1a *.h *.c *.cu 2> /dev/null | xargs astyle --style=java --indent=spaces=2 --align-pointer=name --align-reference=name --unpad-paren --break-closing-brackets --add-brackets --mode=c
done

#eof;;
