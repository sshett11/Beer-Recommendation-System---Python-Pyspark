
--------------------------------INSTRUCTIONS TO EXECUTE ------------------------------------

Python version used - 2.7.15 and keep all the data files in the same folder where you will be executing .py file.

For local server  -
1.) sudo apt-get install python-tk

2.) spark-submit item-item.py
Filename 1 - beer_data.csv
Filename 2 - beer_items.csv
Username - 23203
Other Individual Parameters - 2 1 2 1

3.) python user-user.py
Username - johnmichaelsen
Other Individual Parameters - 2 2 1 2

For Amazon aws or similar (item-item) -
1.) Download zip file of the project
2.) You have two .py files which out of them one is ITEM-BASED-RECOMMENDER(SPARK) & then do changes in code(remove tkinter and take arguments from sys.argv[])
3.) Connect to the cluster and check for pyspark version using command : pyspark
4.) Insert the item-based_recommender.py file in the cluster by using WINSCP drag and drop .
5.) Now insert the data file into the cluster by moving them to edge node usig WINSCP and push them to name node using: hadoop fs -put beer_data.csv /user/amekala/
6.) Follow the same for 3 data files beer_data.csv,beer_items.csv,user_items.csv
7.) Now execute the python file using following command:
spark-submit item-based-recommender.py  {path of the data file beer_data.csv in cluster} {path of the beer_items.csv in cluster} {userid} {preference for aroma}
{preference for appearance} {preference for palate} {preference for taste}
8.) make sure you pass 8 arguments while you execute the command.
9.) user preference can be given either 2 or  1 . where 2 means they prefer, 1 is they don't prefer
10.) user id can be seen in user_items.csv file


-------DATA FILES EXPLAINED-------

beer_data.csv
column1 : overall-rating
column2 : review_aroma
column 3 : review_apperance
column 4 : review_palate
column 5: review_taste
column 6 : user_id
column 7 : beer_id

beer_items.csv
column1: beer_id
columns2: Beernames
