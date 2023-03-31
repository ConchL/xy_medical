#!/bin/bash

for label_name in dead
#for label_name in aki_stage ards dead dic liver_injury shock
do
  for model_type in BiLSTMDayX BiLSTMDayXConstant BiLSTMDayXConstantFC BiLSTMDay5 BiLSTMDay5Constant BiLSTMDay5ConstantFC
  do
    for windows in 1 2 3 4
    do
      for layers in 1 2 3 4
      do
        for hidden_size in 16 32 64 128 256 512
        do
          # python3 -u train.py --model_type $model_type --layers $layers --hidden_size $hidden_size > out_${model_type}_layers${layers}_hs${hidden_size} 2>&1
                                                                                          #'checkpoint/log/{}/{}_{}.log'.format(label_name, label_name, model_type)
    #      nohup python3 -u train.py --label_name $label_name --model_type $model_type --layers $layers --hidden_size $hidden_size --log out_${model_type}_layers${layers}_hidden_size${hidden_size} &
          echo label_name: ${label_name}, model_type: ${model_type}, windows: ${windows}, layers: ${layers}, hidden_size: ${hidden_size}
          python3 -u train_BiLSTM_Dead.py --label_name $label_name --model_type $model_type --windows $windows --layers $layers --hidden_size $hidden_size
        done
      done
    done
  done
done