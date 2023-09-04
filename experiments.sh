# ----- Fold 1
# ---------- ISIC CNN
python3 main.py --experiment isic_cnn_1 --task train --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False;
python3 main.py --experiment isic_cnn_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method none;
python3 main.py --experiment isic_cnn_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_cnn_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- ISIC SWIN
python3 main.py --experiment isic_swin_1 --task train --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True;
python3 main.py --experiment isic_swin_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method none;
python3 main.py --experiment isic_swin_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_swin_1 --task test --seed 123 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 CNN
python3 main.py --experiment sd260_cnn_1 --task train --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False;
python3 main.py --experiment sd260_cnn_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method none;
python3 main.py --experiment sd260_cnn_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_cnn_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 SWIN
python3 main.py --experiment sd260_swin_1 --task train --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True;
python3 main.py --experiment sd260_swin_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method none;
python3 main.py --experiment sd260_swin_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_swin_1 --task test --seed 123 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration True;


# ----- Fold 2
# ---------- ISIC CNN
python3 main.py --experiment isic_cnn_2 --task train --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False;
python3 main.py --experiment isic_cnn_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method none;
python3 main.py --experiment isic_cnn_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_cnn_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- ISIC SWIN
python3 main.py --experiment isic_swin_2 --task train --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True;
python3 main.py --experiment isic_swin_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method none;
python3 main.py --experiment isic_swin_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_swin_2 --task test --seed 456 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 CNN
python3 main.py --experiment sd260_cnn_2 --task train --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False;
python3 main.py --experiment sd260_cnn_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method none;
python3 main.py --experiment sd260_cnn_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_cnn_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 SWIN
python3 main.py --experiment sd260_swin_2 --task train --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True;
python3 main.py --experiment sd260_swin_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method none;
python3 main.py --experiment sd260_swin_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_swin_2 --task test --seed 456 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration True;


# ----- Fold 3
# ---------- ISIC CNN
python3 main.py --experiment isic_cnn_3 --task train --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False;
python3 main.py --experiment isic_cnn_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method none;
python3 main.py --experiment isic_cnn_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_cnn_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- ISIC SWIN
python3 main.py --experiment isic_swin_3 --task train --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True;
python3 main.py --experiment isic_swin_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method none;
python3 main.py --experiment isic_swin_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment isic_swin_3 --task test --seed 789 --dataset ISIC --dataset_dir ../../Datasets/ISIC_2019 --swin_model True --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 CNN
python3 main.py --experiment sd260_cnn_3 --task train --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False;
python3 main.py --experiment sd260_cnn_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method none;
python3 main.py --experiment sd260_cnn_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_cnn_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model False --calibration_method temperature --boundary_calibration True;

# ---------- SD-260 SWIN
python3 main.py --experiment sd260_swin_3 --task train --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True;
python3 main.py --experiment sd260_swin_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method none;
python3 main.py --experiment sd260_swin_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration False;
python3 main.py --experiment sd260_swin_3 --task test --seed 789 --dataset sd260 --dataset_dir ../../Datasets/SD260 --swin_model True --calibration_method temperature --boundary_calibration True;
