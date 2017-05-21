grep -e "\[[ 0-9]*\]\[2\]" $1|sed -e 's/Info: kernel[^=]*=[ ]*[0-9\.]* GFLOPS, max \/ min = \([0-9]*%\)/\1/g' > out.txt
