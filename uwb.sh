#!/bin/sh

rm -f config-*.ini
rm -rf outdir/*

# thread
thread=64

# parameters
N_test=50000
N_train=1200
iteration=600
iteration_svm=50
ratio="5.0 2.0 1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.00001"
CNR_train=10
learningrate_al=0.0001
learningrate_be=0.001
C=1.0
M=16
step_size=0.01
temp_time=4
linewidth=5000000
sam_frequency=10000000000


for rat in $ratio
do
var=${rat}
cp -f config.ini config-ratio${var}.ini
sed -i -e "s/N_test=.*/N_test=${N_test}/g" config-ratio${var}.ini
sed -i -e "s/N_train=.*/N_train=${N_train}/g" config-ratio${var}.ini
sed -i -e "s/iteration=.*/iteration=${iteration}/g" config-ratio${var}.ini
sed -i -e "s/iteration_svm=.*/iteration_svm=${iteration_svm}/g" config-ratio${var}.ini
sed -i -e "s/ratio=.*/ratio=${rat}/g" config-ratio${var}.ini
sed -i -e "s/CNR_train=.*/CNR_train=${CNR_train}/g" config-ratio${var}.ini
sed -i -e "s/learningrate_al=.*/learningrate_al=${learningrate_al}/g" config-ratio${var}.ini
sed -i -e "s/learningrate_be=.*/learningrate_be=${learningrate_be}/g" config-ratio${var}.ini
sed -i -e "s/C=.*/C=${C}/g" config-ratio${var}.ini
sed -i -e "s/M=.*/M=${M}/g" config-ratio${var}.ini
sed -i -e "s/step_size=.*/step_size=${step_size}/g" config-ratio${var}.ini
sed -i -e "s/temp_time=.*/temp_time=${temp_time}/g" config-ratio${var}.ini
sed -i -e "s/linewidth=.*/linewidth=${linewidth}/g" config-ratio${var}.ini
sed -i -e "s/sam_frequency=.*/sam_frequency=${sam_frequency}/g" config-ratio${var}.ini

done

echo start
find . -name 'config-*.ini' | xargs -P ${thread} -IXXX python3 main.py XXX
echo end

rm -f config-*.ini
