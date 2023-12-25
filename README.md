
<html>
  <head>
    <h1>SSML---Spark-Streaming-for-Machine-Learning</h1>
    </head>
  <body>
    <h2>About the project:</h2><h3>Twitter sentiment analysis using spark streaming </h3>
    <h2>Purpose:</h2><h4>The project evaluates tweets to predict their sentiment </h4>
    <h4>Sentiment analysis is extremely useful in social media monitoring as it allows us to gain an overview of the wider public opinion behind certain topics.</h4>
    <h2>Pre-requisites:</h2>
    <h4>->Python 3.8.x</h4>
    <h4>->Hadoop 3.2.2</h4>
    <h4>->Spark 3.1.2</h4>
    <h1>Steps to run the project:</h1>
    <h4>1) Download all the python files into a folder and also create a sentiment folder in it containing test and train data.</h4>
    <h4>2) Open 2 terminals: 1. with path /opt/spark/bin and 2. in the folder where you downloaded the python files </h4>
    <h4>3) run command python3 stream -f sentiment -b <batchsize> in terminal 2</h4>
    <h4>4) run command ./spark-submit sentiment <filepath> in terminal 1</h4>
    <h4>5) Once the models are done with training they get stored in pickle file</h4>
    <h4>6) repeat step 3 and 4 with test data set but with stream1 and sentiment1 files</h4>
  </body>
  
  
 </html>
