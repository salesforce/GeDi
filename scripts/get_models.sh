
cwd=$(pwd)
mkdir ../pretrained_models
cd ../pretrained_models

wget https://storage.googleapis.com/sfr-gedi-data/gedi_topic.zip
unzip gedi_topic.zip
rm gedi_topic.zip



wget https://storage.googleapis.com/sfr-gedi-data/gedi_sentiment.zip
unzip gedi_sentiment.zip
rm gedi_sentiment.zip



wget https://storage.googleapis.com/sfr-gedi-data/gedi_detoxifier.zip
unzip gedi_detoxifier.zip
rm gedi_detoxifier.zip

cd $cwd
