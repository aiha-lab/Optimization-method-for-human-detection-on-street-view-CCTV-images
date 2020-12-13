# Docker Submit Checklist

#### This is Checklist for IITP AI Grand Challenge docker submit.

1. Run docker container from given image. Mount IITP valid dataset path for further mAP test.
'''
sudo nvidia-docker run -it --name "container_name" -v ${DATA_PATH} "image_name"
'''
2. Transfer the model pretrained file(.pt) to docker container environment. 
'''
sudo nvidia-docker cp ${FILE_PATH} "container_name":/aichallenge/weights/
'''

3. Check model's pretrained file name, image size, confidece threshold and batch size.

4. Run mAP test and crosscheck model's mAP.

5. Check there is no json file in /aichallenge/ directory. Otherwise submit error occur.

6. Convert container environment to new docker image.
'''
sudo nvidia-docker commit "container_name" "image_name"
'''
7. Convert docker image to tar file
'''
sudo nvidia-docker save -o "tar_file_name" "image_name"
'''
8. Change tar file's mode for model upload.
'''
sudo chmod 777 "tar_file_name"
'''
9. Check python command works in root mode. If it doesn't work, add alias command in bashrc
'''
vi ~/.bashrc
alias python='root/miniconda3/bin/python'
'''
10. Upload the tar file and prey
	
	



