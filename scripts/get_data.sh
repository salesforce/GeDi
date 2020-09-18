
cwd=$(pwd)

mkdir ../data
cd ../data
mkdir AG-news
cd AG-news

wget https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
wget https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv

cd ../../

python proc_data.py

rm data/AG-news/train.csv
rm data/AG-news/test.csv

cd $cwd
