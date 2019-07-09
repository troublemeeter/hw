for w_alpha in 0.1 1 10 50
do
	for v_alpha in 0.1 1 10 50
	do
		echo 'w_alpha: $w_alpha, v_alpha: $v_alpha'
		cat ./train.libsvm|\
		./bin/fm_train_softmax \
		-m bigmodel \
		-cn 6 \
		-dim 1,1,50 \
		-w_l1 0.1 \
		-w_l2 0.1 \
		-v_l1 0.1 \
		-v_l2 0.1 \
		-init_stdev 0.1 \
		-w_alpha $w_alpha \
		-v_alpha $v_alpha \
		-core 10

		cat ./train.libsvm|./bin/fm_predict_softmax \
		-m bigmodel -cn 6 -dim 50 -out bin/train.output -core 10

		cat ./test.libsvm|./bin/fm_predict_softmax \
		-m bigmodel -cn 6 -dim 50 -out bin/test.output -core 10
		
		python eval.py
	done
done