<h2 align= "center"><em>Personalized Cancer Diagnosis</em></h2>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/Personalized-Cancer-Diagnosis/blob/main/src%20files/LEADNEW.gif"/>
</div>

<hr width="100%" size="2">

<h3 align= "left"> <b> Key Project Formulation </b> </h3>

<br>

<p>
<strong>Real World/Business Objective :</strong> Classify the given genetic variations/mutations based on evidence from text-based clinical literature.
</p>

<br>

<p>
<strong>Constraints :</strong>
</p>
<ol>
<li>Interpretability </li>
<li>Probability of a data-point belonging to each class is needed</li>
<li>No low-latency requirement</li>
<li>High Precision (Errors can be very costly)</li>
</ol>

<br>

<p>
<strong>Get the data from :</strong> https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
<br>The data is provided by kaggle as Research Prediction Competition.
</p>

<br>

<p>
<strong>Data Overview :</strong>
<br>
  <p>We have two data files: one conatins the information about the genetic mutations and the other contains the clinical evidence (text) that  human experts/pathologists use to classify the genetic mutations.Both these data files are have a common column called ID</p>
<br>
<p>
    Data file's information:
    <ul>
        <li>
        training_variants (ID , Gene, Variations, Class)
        </li>
        <li>
        training_text (ID, Text)
        </li>
    </ul>
</p>

<br>

<br />


<p>
<strong>ML Problem Formulation :</strong>
</p>
<p> <strong>There are nine different classes a genetic mutation can be classified into => Multi class classification problem</strong> </p>
<br>
<p>Objective is to Predict the probability of each data-point belonging to each of the nine classes.</p>

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Multi class log-loss</li>
<li>Confusion Matrix</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Flow of Project : </h3>
  
  <br>

  <h3 align= "center"><strong>Exploratory Data Analysis</strong></h3>
  <p align= "center"><em> - Basic text preprocessing like removal of stopwords, removing extra spaces, lower casing text etc. </em></p>
  <p align= "center"><em> - test, train and cross validation split </em></p>
  <p align= "center"><em> - Univariate analysis </em></p>
  
  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center"><strong>Prediction using Random Model </strong></h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Building Machine Learning Models</h3>
  <p align= "center"><em> - Baseline model is naive bayes hyperparameter tuned model  </em></p>
  <p align= "center"><em> - K Nearest Neighbhor with hyperparameter tuning   </em></p>
  <p align= "center"><em> - Logistic regression with and without class balancing and hyperparameter tuning  </em></p>
  <p align= "center"><em> - Linear support vector machine with hyperparameter tuning </em></p> 
  <p align= "center"><em> - Random forest classifier with hyperparameter tuning </em></p>  
  <p align= "center"><em> - Linear support vector machine with hyperparameter tuning and one hot encoding as well as response coding </em></p>  
  <p align= "center"><em> - Stacking classifier </em></p>  
  <p align= "center"><em> - Maximum voting classifier </em></p>  
  <p align= "center"><em> - Feature importance from all above models </em></p>  



  
</body>

<hr width="100%" size="2">
<br>

<p>
<strong>Confusion matrices of the best model :</strong>
</p>
<div align="center">
  <img height="200" src="https://github.com/shreyjain99/Quora-Question-Pair-Similarity/blob/main/src%20files/image.png"/>
</div>

<br>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Try out models (Logistic regression, Linear-SVM) with simple TF-IDF vectors instead of TD_IDF weighted word2Vec. </li>
<li>Perform hyperparameter tuning of XgBoost models using RandomsearchCV with vectorizer as TF-IDF W2V to reduce the log-loss.</li>
</ol>

<hr width="100%" size="2">
