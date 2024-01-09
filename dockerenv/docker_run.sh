# parent directory of /data and /models
DATAPATH = C:/Users/Eddie/Documents/DurcanLab/HealthyUnhealthyClassifier/data_models


# these two lines create docker file
docker build --rm -f dockerfile -t healthyunhealthyclassifier .
docker run --rm -di -p 6006:6006 -v $DATAPATH:/volume -p 8888:8888 -env-file --name huccontainer healthyunhealthyclassifier:latest 

docker exec huccontainer  sh -c "bash /developer/run_cnn.sh"