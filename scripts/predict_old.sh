echo Predict Start........;
python predict_old.py /raid/iitpdata/images/val/;
echo Predict Done.........;
echo IITP mAP caclulate Start.........;
python 4-4_calculate.py /raid/iitpdata/converted.json t4_res_U0000000221.json;
echo IITP mAP caclulate Done.........;
echo Remove all json file;
rm *.json

