#!/bin/bash

if [ $# -ne 1 ] ; then
  echo "The first parameter is the name of the file to be formatted"
  echo "Example: $0 hello_world.c"
  exit 0
fi

astyle --style=java --indent=spaces=2 --align-pointer=name --align-reference=name --unpad-paren --break-closing-brackets --add-brackets --mode=c $1

#eof;;
