cd dataset/custom/

rm gt/* 
rm gt.zip
rm submit/*
rm submit.zip

cp $1/gt/*.txt gt
cd gt/;zip -r  gt.zip * 
mv gt.zip ../
cd ../

cp $1/*.txt submit
cd submit/;zip -r  submit.zip * 
mv submit.zip ../
cd ../
python Evaluation_Protocol/script.py -g=gt.zip -s=submit.zip
