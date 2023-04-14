# 999 257 33333 1234 9876 1045 32447 4377489 33333333 2147483645
for n in 100 1000 10000 100000 1000000 10000000 100000000
do
	echo "running: $n"
	bash ./run.sh ./odd_even_sort $n ./data/$n.dat > "my_log/run_$n.log"
done