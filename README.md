# Favorite Song Tournament
## 1. Motivation
If you love listening to music, it can sometimes be very hard to answer a simple question: what is your favorite song? Many people don't necessarily have a single favorite song, but rather a playlist of songs they enjoy listening to. In fact, people's song tastes naturally change over time, so they may end up making a new favorite playlist every so often, capturing the essence of their taste in that moment. Yet, the same problem remains: how do we articulate what our taste in music is? Genre and artists can only take you so far, as the number of popular music genres and fusions between them have skyrocketed over the years, and artists feel pressure to reinvent themselves with every new release to keep their craft in style. 

But as I alluded to earlier, a playlist can capture the essence of your taste at a certain time; we just need to extract the right kind of data from it. What if you could use data to finally answer that lingering, nagging question: what is your favorite song? I develop an application that helps users analyze a playlist and discern what specific qualities they like about the songs they listen to. It will suggest musical/lyrical features used to compare the songs in the playlist, or accept user suggestions if they already have ideas. Users will have the opportunity to collect and import data themselves, generate data interactively within the application, or use no data at all, according to their personal preference for effort invested into the process. Lastly, the tournament will help them find their favorite song of the moment, rank all of the songs in the playlist, and comment on the demonstrated user preference for each input feature used in those decisions.

In short, I want to build an application that encourages people curious about their music taste to think deeply about what they enjoy in the songs they listen to, so they can start finding new music that fits the vibe they are looking for. 

## 2. Software Goals
### a) Functionality
The application will first take in a playlist (exact format depends on stage of development) to ensure that the number of songs in the tournament is between 8 and 128. These numbers were chosen partially for convenience (they are powers of 2, very nice numbers for the Swiss tournament system), and also the practical insights the tournament results will provide. A playlist larger than 128 songs (7-8 hours end to end) likely doesn't define the user's niche tastes very well, while a playlist smaller than 8 songs (about 30 min) probably doesn't have enough information to accomplish that goal. 

Once the playlist size has been verified, the user will be prompted to choose a tournament format (see "3. Tournament Formats" below). Each format requires a certain level of front-end work by the user, ranging from no initial work to requesting that the user upload a dataset. Regardless of the tournament type, the user is also prompted to select the criteria under which they want to rate the songs. They have the option to choose from a list of suggestions, or import a list of their own criteria if they already have ideas. Lastly, they will be asked to select a match size from 2-8, which is the maximum number of songs they will be asked to compare at a time. These numbers were chosen to provide a spectrum between ease of decision making (comparing only 2) and the speed of the tournament (comparing 8 at a time gives a lot of information to rank the songs with minimal tournament rounds). 

After the user has selected the parameters of the tournament specified above, they can run the tournament. The amount of work they will need to do at this stage is inversely proportional to the front-end work they choose to do when setting up the tournament. If they provided their own dataset, they will be able to teach a neural network to make decisions for them (manually enter one round of the tournament, and then the neural network learns correlations between match outcomes and features). If they didn't want to use any data, they will manually run the tournament (the application will display one match at a time and ask for the outcomes). 

Upon completion of all the matches, the user will be able to view the results. This may include a full ranked list of songs in the playlist, a subset of noteworthy songs that scored the highest, or the sole winner (the choice of output depends on user preference and the specific tournament format, see below). Additionally, the user will get to see how each input feature contributed to the rankings, providing descriptive insights into their taste in music. 

### b) Stages Of Development
**Phase 1: Console Output**
The first phase of this project is a proof of concept where I demonstrate that all tournament algorithms provide meaningful insights before focusing on the logistics of enhancing the user interface. In this prototype, the user will enter a playlist size into the console. 

If the number of songs is within the valid range (8-128), the user will then input their choice of tournament format into the console. Based on that choice, the program may ask them for additional data (see "3. Tournament Formats" below). If that additional data is properly formatted (contains the correct number of songs, has no missing data). This data will contain the features chosen by the user (either from a list of suggestions from the program or metrics the user wants to test). Then, the user inputs a match size to the console.

During the actual tournament execution, each match the user is asked to provide an outcome will be printed to the console, and the user will be notified when all automated match results have been decided. Then, the full results (rankings and comments on how each feature contributed to them) will be printed to the console.

**Phase 2: File Output** 
The only change made from phase 1 in this next phase is that the output will be a text or markdown file (.txt, .md, or .pdf). This gives the user something to keep from each run of the program. 

**Phase 3: Python Executable File (.exe)** 
In Phase 3, I want to create a nice GUI and make it easier for the user to run the application. Instead of having to clone the repository and run driver code in Python, they can just download and run an executable file (.exe). When the file starts running, the user is prompted to enter the number of songs in their playlist in a dialog box. Then, the screen is divided into sections (buttons) that each represent a tournament format, and a textbox explaining them. After a selection is made, the program will ask the user to upload a .xlsx file with more data if necessary. Then, another dialog box appears to let the user choose a match size. 

Once the tournament is set up, the match window opens. The user is shown a match that they have to call, which feeds information to the program. Once the user completes all of the matches they are prompted to judge, the program will automate the rest of the tournament (if it is unfinished). Then, the results will be displayed on the screen. The user can select which subset of the rankings they want to see (one, notable i.e. top N, all). One section of the screen will have the ranked list, another will prominently show the winner of the tournament, and the last section will provide insights about how each feature contributed to the rankings (correlation between feature score and final score in the tournament). The user will be able to click a button and download the results in a format they choose (.txt, .md, .pdf). 

**Phase 4: Create Application With Spotify API** 
My ultimate vision for this project is to be able to run the application directly from Spotify or from a webpage that can access a Spotify playlist. This would make the input very clean and easy from the user's perspective (just click a button from the page of that playlist or import the playlist to an external site). Once the size of the playlist is verified to be compatible with the program, the application will launch in another tab or window, where they can run the tournament and get output as they can in phase 3. 

## 3. Tournament Formats
### a) Swiss (No Data Needed)
If the user doesn't want to provide any initial data, they can simply call all of the matches manually. Matches will be presented to them using the Swiss tournament structure, which pools participants based on their current score in the tournament. At the start, each song will have a score of 0, and they will get one point for every song they beat in a match. For instance, if the match size is 2, the winner of each match gets 1 point added to their score. If the match size is 8, the songs will be ranked 1-8 by the user. First place will get 7 points (beat out 7 other songs), second place will get 6, and so on. When a round is over, the songs are sorted by their score, greatest to least, and the next round's matches are consecutive pairs in that ranked list (1 v.s. 2, 3 v.s. 4, etc). 

If the user chooses a match size that is not a divisor of the total number of songs in the tournament (for example, a match size of 2 for a playlist of 15), then the song list will be padded with dummies to round up to a multiple of the match size. Dummies will automatically lose all matches it is placed in (obtain the lowest ranking possible as to not affect the rankings of the real songs). The user will never see a dummy song (if they are prompted to call a match involving a dummy, the dummy is excluded from the display and automatically given last place in that match). 

The tournament will end when, at the conclusion of a round, one song has a unique winning score. The songs will be ranked based on final scores, and tie breakers will be based on strength of schedule. If there are still any ties left after strength of schedule analysis, rankings between those songs will be randomly chosen. 

One limitation of this method is that if song A beats song B in a match, song A is not guaranteed to have a better final ranking than song B. This may happen because real world preferences don't always follow the transitive property: if song A beats song B, and song B beats song C, then song A must beat song C. Depending on the outcomes of each match, it may not even be mathematically/logically possible to enforce a rule that guarantees that song A will have a higher final ranking than song B. 

### b) Sorting (Absolute Feature Data Required)
When I say "absolute feature data", I mean that each feature chosen to judge the songs by has a grading scale and rubric, and the user's score for each song is chosen independently of the scores of the other songs. In other words, this score will be the same if the song is moved into another playlist. 

If the user uploads a dataset with absolute feature data, the program can leverage a Deep Q Network (DQN) to learn how feature scores contribute to match outcomes. This learning happens after the user manually completes the first round of the tournament.

I define a "sorting" tournament structure using the following algorithm:
1. Assign each song a score of 0
2. Choose (match size - 1) songs at a random as a separators
3. Each match in the first round is the separators plus one other song, until every song has faced the separators
4. This process "sorts" the songs into pools based on how many of the separators they each beat
5. Notice that we automatically know the final rankings of the separators based on the size of each pool
6. Repeat steps 2-5 for each pool, until the pool size is less than or equal to the match size (in which case 1 match can rank that pool)
7. The tournament ends when there are no more pools with a size greater than 1

The user manually provides the results for each match in the first round, training the DQN to analyze how each feature in the dataset contributed to those calls. Then, the DQN will complete the tournament automatically. Unlike in Swiss tournament, if song A beats song B in any match in any round, it is guaranteed to have a higher ranking at the end because it will be moved up into a higher pool. However, one limitation of this format is that the DQN may not accurately reflect user preferences.

### c) Hybrid (Relative Feature Data Required)
When I say "relative feature data", I mean that songs are graded on each feature according the following algorithm:
1. Choose a feature to grade. Give each song a score of 1.
2. At most half of the songs are upgraded to 2.
3. At most half of the songs with score 2 are upgraded to score 3, and so on until one song has the unique high score for that feature.
4. Repeat steps 1-3 for all features.
5. Any song that is strictly dominated by another song (has a strictly lower feature score for all features) is eliminated from the tournament and not displayed in the final rankings.

This is relative data because a song's score also depends on the strength of all the other songs in the tournament, so it could differ depending on the playlist it is put into. After the dropout of strictly dominated songs, the user has the option to start a smaller Swiss or Sorting tournament with the songs that remain. 

Since the user is in control of the relative data, they have more influence over the final result than in the Sorting model. However, the drawback of this method is the extra effort required to get the relative feature data. The user does not need to provide this data before the tournament is run, because the program will solicit input to generate that data. Much like feeding the user matches in the Swiss tournament algorithm, the hybrid algorithm will give the user batches of the song data in which they have to choose at most half to "upgrade" (add 1 to the current score for the current feature). This will happen iteratively until all the relative feature data is generated. The risk here is that not all batches will be a good mix of songs the user would upgrade or keep the same. The user has to make a judgement call: if they care about seeing the top three, they choose a batch size of six to guarantee that the top three will rise to the top. If the chosen batch size does not divide the total number of songs, dummy songs are added to pad the data. The dummy songs will be distributed evenly among the batches and automatically not advance. 

## 4. Codebase Architecture
### Phase 1

## 5. Installation & Usage
### Phase 1
