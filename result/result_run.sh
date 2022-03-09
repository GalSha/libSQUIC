export GAMMA=2.0;python3 ../test.py > temp.log
grep out_ temp.log > results_2.m

export GAMMA=1.5;python3 ../test.py > temp.log
grep out_ temp.log > results_15.m

export GAMMA=1.2;python3 ../test.py > temp.log
grep out_ temp.log > results_12.m

export GAMMA=1.0;python3 ../test.py > temp.log
grep out_ temp.log > results_1.m

export GAMMA=0.5;python3 ../test.py > temp.log
grep out_ temp.log > results_05.m

export GAMMA=0.0;python3 ../test.py > temp.log
grep out_ temp.log > results_0.m