# TIN175
Predicting temperature of city with weather data from nearby cities



<H3> Plan and suggested contributors</H3>
1. Choose dataset finding important variablels (Jithinraj, André)\n
2. Visualize dataset (André)<br>
3. Preprocess the dataset (Felix) <br>
4. Add recurrent network from keras to model (Edin) <br>
5. Search for hyperparameters and network architecture (Rasmus)<br>
6. discuss the results (Everyone) <br>
7. Strucute of essay(Edin)

<br><br><br><br><br><br>
Things to be tested <br>
1. Find variables that are important with regression<br>
2. Try adding cities further away<br>
3. Using different inputs i.e. <temp, windspeed, winddir> , <temp, percipitation, humidity, winddir> <br> to predict <temp, windspeed, winddir> , <temp, percipitation><br>
4. Try at different training time. Do we need data from 40 years back or 20years back<br>
5. Compare LSTM and GRU network performance<br>


<u>André</u> <br>\<br>

Found and downloaded good csv files that we could combine to a dataset<br><br>

Preprocessing<br>
- Visualization of datasets to see which ones are good<br>
- Searched and removed faulty values<br><br>

Training<br>
- Simulations/training with LSTM, GRU<br><br>

Report<br>
- Background<br>
- Further work<br><br><br>


<u>Felix</u><br><br>

Preprocessing
- Script to merge csv files
- Liner interpolation

Training<br>
- Simulations/Training with LSTM, GRU <br>
- Hyperparameter testing<br>
- Plotting results with Tensorboard<br>\<br>

Report<br>
- Introduction<br>
- Background<br>
- Results<br><br><br>


<u> Rasmus </u><br><br>
Training<br>
- Simulations/Training with LSTM, GRU <br>
- Hyperparameter testing<br>
- Plotting results with Tensorboard<br><br>

Report<br>
- Method<br><br>


<u> Edin </u> <br><br>
Preprocessing
- RNN network to replace missing data in dataset <br><br>

Training<br>
- Simulations/Training with LSTM, GRU <br><br>


<u> Jithin </u><br><br>
Preprocessing <br>
- What variables are the most important with linear regression <br><br>

Training<br>
- Simulations/Training with LSTM, GRU <br><br>

Report<br>
- Discussion














<br><br><br><br><br><br>


<table class="tg">
  <tr>
    <th class="tg-0lax"><span style="font-weight:bold">Members</span></th>
  </tr>
  <tr>
    <td class="tg-0lax">Rasmus Claesen</td>
  </tr>
  <tr>
    <td class="tg-0lax">Felix</td>
  </tr>
  <tr>
    <td class="tg-0lax">Edin</td>
  </tr>
  <tr>
    <td class="tg-0lax">Jithinraj Sreekumar</td>
  </tr>
  <tr>
    <td class="tg-0lax">André Höjmark</td>
  </tr>
</table>
