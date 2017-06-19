Open Bash shell
Go to directory with aws key pair - Downloads aws_key_pair.pem
chmod 400 aws_key_pair.pem
Open AWS and start EC2 instance, get public IP address
# Load files
scp -i "aws_key_pair.pem" /c/Users/rickdberg/Documents/Git/machine_learning/site_metadata_compiler_completed.py ubuntu@54.187.49.191:~/site_metadata_compiler_completed.py
scp -i "aws_key_pair.pem" /c/Users/rickdberg/Documents/Git/machine_learning/idw_interp.py ubuntu@54.187.49.191:~/idw_interp.py


ssh -i "aws_key_pair.pem" ubuntu@54.187.49.191
sudo apt install python3
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
exit

ssh -i "aws_key_pair.pem" ubuntu@54.187.114.172
conda install numpy
conda install -c conda-forge geopy=1.11.0
conda install pandas
conda install sqlalchemy
conda install rasterio
conda install -c anaconda mysql-connector-python=2.0.4

ssh -L 3307:69.91.147.103:3306 \
ubuntu@54.187.49.191
