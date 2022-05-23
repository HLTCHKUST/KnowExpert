echo "Prepare WoW data..."

mkdir data/wizard_of_wikipedia
cd data/wizard_of_wikipedia
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar zxvf wizard_of_wikipedia.tgz


echo "Prepare CMU DoG data..."
cd ../data
git clone https://github.com/festvox/datasets-CMU_DoG

unzip data/ITDD_data.zip -d data/datasets-CMU_DoG
