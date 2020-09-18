pip install transformers==2.8
pip install -r ../hf_requirements.txt

git clone https://github.com/NVIDIA/apex
#Comment apex installation below if fp16 isn't required
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

apt-get update
apt-get install wget
apt-get install unzip

# transformers installation from source
# git clone https://github.com/huggingface/transformers
# cd transformers
# git checkout 21da895013a95e60df645b7d6b95f4a38f604759
# pip install .
# pip install -r examples/requirements.txt
# cd ..