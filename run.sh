# 改成linux 换行符

mkdir dataset
cd dataset
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/age_train.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/age_test.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/user_basic_info.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/user_behavior_info.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/user_app_actived.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/app_info.csv
wget -c https://digix-algo-challenge.obs.cn-east-2.myhuaweicloud.com/demographic_predicton_data/user_app_usage.tar.bz2
tar -xjf user_app_usage.tar.bz2.tar.bz2
rm user_app_usage.tar.bz2.tar.bz2

python prepare.py

cd ..
python train.py