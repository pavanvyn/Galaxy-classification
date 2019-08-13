# create folder for experimental output
mkdir log/galTest

# change to source directory
cd src

# run experiment on terminal
python main.py ../log/galBarred/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 0;

python main.py ../log/galElliptical/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 1;

python main.py ../log/galSpiral/ ../data/train_test.csv ../data/apply_model.csv --objective one-class --train True --lr 0.0001 --n_epochs 100 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 100 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --apply_model True --normal_class 2;

# copy following to load previous model
--load_config ../log/galBarred/config.json --load_model ../log/galBarred/model.tar
--load_config ../log/galElliptical/config.json --load_model ../log/galElliptical/model.tar
--load_config ../log/galSpiral/config.json --load_model ../log/galSpiral/model.tar
