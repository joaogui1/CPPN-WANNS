wann_center_train:
	nohup python3 wann_train.py -p p/evosoro.json -n 30 -o wann_center > wann_center.txt &

wann_nocenter_train:
	nohup python3 wann_train.py -p p/evosoro.json -n 30 -o wann_nocenter > wann_nocenter.txt &

wann_recur_train:
	nohup python3 wann_train.py -p p/evosoro.json -n 30 -o wann_recur > wann_recur.txt &

neat_center_train:
	nohup python3 neat_train.py -p p/evosoro.json -n 30 -o neat_center > neat_center.txt &

neat_nocenter_train:
	nohup python3 neat_train.py -p p/evosoro.json -n 30 -o neat_nocenter > neat_nocenter.txt &

neat_recur_train:
	nohup python3 neat_train.py -p p/evosoro.json -n 30 -o neat_recur > neat_recur.txt &

clean:
	rm prettyNeatWann/*.vxa