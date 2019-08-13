# Galaxy-classification

#This algorithm is derived from https://github.com/lukasruff/Deep-SVDD, so please refer to that.

#Please check the various input parameters in the main.py module
#Running the code where training/testing data are listed in data/train_test.csv and application data in data/apply_model.csv

#normal class 0 (Barred) - logfile log/galBarred
python main.py ../log/galBarred/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 0;

#normal class 1 (Elliptical) - logfile log/galElliptical
python main.py ../log/galElliptical/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 1;

#normal class 2 (Spiral) - logfile log/galSpiral
python main.py ../log/galSpiral/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 2;
