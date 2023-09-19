# ----- Fold 1
# ---------- ISIC CNN
python3 main.py --experiment isic_cnn --task train --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model False;
python3 main.py --experiment isic_cnn --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model False --calibration_method none;
python3 main.py --experiment isic_cnn --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_cnn --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model False --calibration_method temperature --boundary_calibration True;

# ---------- ISIC ResNet
python3 main.py --experiment isic_resnet --task train --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model True;
python3 main.py --experiment isic_resnet --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model True --calibration_method none;
python3 main.py --experiment isic_resnet --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_resnet --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --resnet_model True --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 CNN
python3 main.py --experiment sd260_cnn --task train --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model False;
python3 main.py --experiment sd260_cnn --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model False --calibration_method none;
python3 main.py --experiment sd260_cnn --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_cnn --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model False --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 ResNet
python3 main.py --experiment sd260_resnet --task train --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model True;
python3 main.py --experiment sd260_resnet --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model True --calibration_method none;
python3 main.py --experiment sd260_resnet --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_resnet --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --resnet_model True --calibration_method temperature --boundary_calibration True;
