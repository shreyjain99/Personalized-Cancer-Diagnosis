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


<p>
<strong>Features in the dataset :</strong>
</p>
<pre>
<b>Id</b> - the id of a training set question pair<br />
<b>qid1, qid2</b> - unique ids of each question (only available in train.csv)<br />
<b>question1, question2</b> - the full text of each question<br />
<b>is_duplicate</b> - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.<br />
</pre>

<br />

<br>

<p>
<strong>ML Problem Formulation :</strong>
</p>
<p> <strong>It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.</strong> </p>

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>log-loss </li>
<li>Binary Confusion Matrix</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Flow of Project : </h3>
  
  <br>

  <h3 align= "center"><strong>Reading Data and Basic Stats</strong></h3>
  <p align= "center"><em> - Total number of unique questions </em></p>
  <p align= "center"><em> - Checking for Duplicates </em></p>
  <p align= "center"><em> - Number of occurrences of each question </em></p>
  <p align= "center"><em> - Checking for NULL values </em></p>
  
  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center"><strong>Basic Feature Extraction (before cleaning) </strong></h3>
  <p align= "center"><em> <strong> Constructing new features like :  </strong></em></p>
  <p align= "center"><em> 1. freq_qid1 = Frequency of qid1's </em></p>
  <p align= "center"><em> 2. freq_qid2 = Frequency of qid2's </em></p>
  <p align= "center"><em> 3. q1len = Length of q1 </em></p>
  <p align= "center"><em> 4. q2len = Length of q2 </em></p>
  <p align= "center"><em> 5. q1_n_words = Number of words in Question 1 </em></p>
  <p align= "center"><em> 6. q2_n_words = Number of words in Question 2 </em></p>
  <p align= "center"><em> 7. word_Common = Number of common unique words in Question 1 and Question 2 </em></p>
  <p align= "center"><em> 8. word_Total = Total num of words in Question 1 + Total num of words in Question 2 </em></p>
  <p align= "center"><em> 9. word_share = (word_common)/(word_Total) </em></p>
  <p align= "center"><em> 10. freq_q1+freq_q2 = sum total of frequency of qid1 and qid2 </em></p>
  <p align= "center"><em> 11. freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2 </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Preprocessing of text (Natural Language Processing)</h3>
  <p align= "center"><em> - Removing html tags </em></p>
  <p align= "center"><em> - Removing Punctuations </em></p>
  <p align= "center"><em> - Performing stemming </em></p>
  <p align= "center"><em> - Removing Stopwords </em></p>
  <p align= "center"><em> - Expanding contractions etc. </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Advanced Feature Extraction (NLP and Fuzzy Features)</h3>


  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Visualization of Data</h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Featurizing text data with tfidf weighted word-vectors </h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Building Machine Learning Models</h3>
  <p align= "center"><em> - Building a random model (Finding worst-case log-loss)  </em></p>
  <p align= "center"><em> - Logistic Regression with hyperparameter tuning   </em></p>
  <p align= "center"><em> - Linear SVM with hyperparameter tuning  </em></p>
  <p align= "center"><em> -  XGBoost </em></p>  



  
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
