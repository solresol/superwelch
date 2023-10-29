Superwelch Demo
===============

Requires an Elgate Stream Deck XL, otherwise it's really boring.


1. `cd cli`

2. `./synthesize.py --sql ~/.bmap-demo-model.sqlite --progress`

3. `./bmap3_train.py --sql  ~/.bmap-demo-model.sqlite`

4. Make sure the stream deck is configured. There are backups of:
  
   - Welch Game
   
   - Welch Game Control 1
   
   - Welch Game Control 2
   
   - Welch Game Control 3   
   
   - Welch Game Experiment 1
   
   - Welch Game Experiment 2
   
   - Welch Game Experiment 3  
   
   - Welch Calculate Results
   
   - Welch Transition
   
The Welch Demo plugin should install 8 other profiles, and two actions:

 - Test results (which will jump to one of the first 4 plug-in-specific profiles)
 
 - Lolly jump (which will jump to one of the second 4 plug-in-specific profiles)
 
5. `./bmap3_infer.py --model ~/.bmap-demo-model.sqlite  --session ~/.bmap-demo-session.sqlite`

(Leave that last program open. Make sure that the focus is directed to it.)



## bmap3_infer.py

It has a little command language:

 - **new** start a new experiment
 
 - **control1=1** set the value of control value 1 to be 1
 
 - **experimental2=3** set the value of the second experimental value to be 3
 
 - **test** run the BMAP3 and Welch tests
 
 - **failed** tell the system it was a failed experiment
 
 - **succeeeded** tell the system it was a successful experiment
 
 new
 control1=1
 control2=5
 control3=3
 experimental1=1
 experimental2=2
 experimental3=4
 test
 failed
 success 



