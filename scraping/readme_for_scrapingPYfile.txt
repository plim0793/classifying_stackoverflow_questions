Scroll to the very bottom (line 158ish) where the scrape_user function is called.
You will need to change:
range(19, 248):
  change it to range(250)
  I had it differently because I messed up too many times
858 + counter * 400:
  this is the starting page to scrape of each loop
  assuming counter == 0, then
  change this to reflect the very first page you are scraping
858 + (counter + 1) * 400:
  this is the ending page of each loop
  this should be identical to the previous parameter except for
  (counter + 1)
r'users_' + str(counter + 3) + r'.csv':
  this is the csv file name to save as
  assuming counter == 0, and I want to save to users_1.csv
  then I do r'users_' + str(counter + 1) + r'.csv'
  I happened to have (counter + 3) because I messed up too many times


To call the .py file,
in the AWS terminal
cd to when this file is saved at,
then type
sudo chmod 777 mcnulty_scrape.py
then type
./mcnulty_scrape.py > mcnulty.log

Any output will be saved to a text file called mcnulty.log in the same folder.
I've disabled most of the printout, so most likely the log file will be empty.

After the script starts running,
in the same folder,
you can check the csv files by typing:
ls -l user* ; wc -l user* 
