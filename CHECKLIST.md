# Docker Submit Checklist

#### This is Checklist for IITP AI Grand Challenge docker submit.

***

1. Run docker container from given image. Mount IITP valid dataset path for further mAP test.
<pre>
<code>
sudo nvidia-docker run -it --name "container_name" -v ${DATA_PATH} "image_name"
</code>
</pre>
2. Transfer the model pretrained file(.pt) to docker container environment. 
<pre>
<code>
sudo nvidia-docker cp ${FILE_PATH} "container_name":/aichallenge/weights/
</code>
</pre>
3. Check model's pretrained file name, image size, confidece threshold and batch size.


4. Run mAP test and crosscheck model's mAP.


5. Check there is no json file in /aichallenge/ directory. **Otherwise submit error occur.**


6. Convert container environment to new docker image.
<pre>
<code>
sudo nvidia-docker commit "container_name" "image_name"
</code>
</pre>
7. Convert docker image to tar file
<pre>
<code>
sudo nvidia-docker save -o "tar_file_name" "image_name"
</code>
</pre>
8. Change tar file's mode for model upload.
<pre>
<code>
sudo chmod 777 "tar_file_name"
</code>
</pre>
9. Check python command works in root mode. If it doesn't work, add alias command in bashrc
<pre>
<code>
vi ~/.bashrc
alias python='root/miniconda3/bin/python'
</code>
</pre>
10. Upload the tar file
	
	



