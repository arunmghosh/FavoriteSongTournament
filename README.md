# Favorite Song Tournament
## 1. Motivation
If you love listening to music, it can sometimes be very hard to answer a simple question: what is your favorite song? Many people don't necessarily have a single favorite song, but rather a playlist of songs they enjoy listening to. In fact, people's song tastes naturally change over time, so they may end up making a new favorite playlist every so often, capturing the essence of their taste in that moment. Yet, the same problem remains: how do we articulate what our taste in music is? Genre and artists can only take you so far, as the number of popular music genres and fusions between them have skyrocketed over the years, and artists feel pressure to reinvent themselves with every new release to keep their craft in style. 

But as I alluded to earlier, a playlist can capture the essence of your taste at a certain time; we just need to extract the right kind of data from it. What if you could use data to finally answer that lingering, nagging question: what is your favorite song? I develop an application that helps users analyze a playlist and discern what specific qualities they like about the songs they listen to. It will suggest musical/lyrical features used to compare the songs in the playlist, or accept user suggestions if they already have ideas. Users will have the opportunity to collect and import data themselves, generate data interactively within the application, or use no data at all, according to their personal preference for effort invested into the process. Lastly, the tournament will help them find their favorite song of the moment, rank all of the songs in the playlist, and comment on the demonstrated user preference for each input feature used in those decisions.

In short, I want to build an application that encourages people curious about their music taste to think deeply about what they enjoy in the songs they listen to, so they can start finding new music that fits the vibe they are looking for. 

## 2. Software Goals
### a) Functionality
The application will first take in a playlist (exact format depends on stage of development) to ensure that the number of songs in the tournament is between 8 and 128. These numbers were chosen partially for convenience (they are powers of 2, very nice numbers for the Swiss tournament system), and also the practical insights the tournament results will provide. A playlist larger than 128 songs (7-8 hours end to end) likely doesn't define the user's niche tastes very well, while a playlist smaller than 8 songs (about 30 min) probably doesn't have enough information to accomplish that goal. 

Once the playlist size has been verified, the user will be prompted to choose a tournament format (see "3. Tournament Formats" below). Some formats require data, while others don't. Regardless of the tournament type, the user is also prompted to select the criteria under which they want to rate the songs. They have the option to define custom criteria, choose from a list of suggestions, or use the features of an existing dataset. Lastly, they will be asked to select a match size from 2-8, which is the maximum number of songs they will be asked to compare at a time. These numbers were chosen to provide a spectrum between ease of decision making (comparing only 2) and the speed of the tournament (comparing 8 at a time gives a lot of information to rank the songs with minimal tournament rounds). 

After the user has selected the parameters of the tournament specified above, they can run the tournament. The amount of work they will need to do at this stage is inversely proportional to the front-end work they choose to do when setting up the tournament. If they provided their own dataset, they will be able to teach a neural network to make decisions for them (manually enter one round of the tournament, and then the neural network learns correlations between match outcomes and features). If they didn't want to use any data, they will manually run the tournament (the application will display one match at a time and ask for the outcomes). 

Upon completion of all the matches, the user will be able to view the results. This may include a full ranked list of songs in the playlist, a subset of noteworthy songs that scored the highest, or the sole winner (the choice of output depends on user preference and the specific tournament format, see below). Additionally, the user will get to see how each input feature contributed to the rankings, providing descriptive insights into their taste in music. 

### b) Stages Of Development
**Phase 1: Console Output**
The first phase of this project is a proof of concept where I demonstrate that all tournament algorithms provide meaningful insights before focusing on the logistics of enhancing the user interface. In this prototype, the user will enter input into the console. 

The user is first asked if they want to use an existing (cached) dataset, or create a new one. If the user chooses to upload a cached dataset, they can choose any dataset already present in the data folder. Otherwise, they are asked to provide the path to the input file (spreadsheet) containing the list of songs in the playlist. This spreadsheet must contain song titles and artist labels, but can also contain data on other song features. However, it must contain 8-128 songs (anything outside that range is invalid). 

In either case, the next prompt is whether or not they want to use just song names (title and artist), absolute feature data, or relative feature data. Absolute feature data contains scores based on a fixed rubric, and each song's score is independent of the others. By contrast, relative feature data describes how each song performs on each feature relative to the other songs in the playlist. If a dataset has only song titles and artist labels, the user is asked if they want to create additional feature data or use it as is. If they want to create additional data, they specify the criteria they want, and whether they want to create absolute or relative feature data.

The program guides the user step-by-step through creating new data. For absolute feature data, the user constructs a rubric for each feature (silently standardized/normalized by the program later so the features all have the same scale), and assigns a value for each song using that rubric. For relative data, all songs start with score 1 and at most half are promoted to score 2. At most half of those are promoted to score 3, and so on until there is a sole highest score. In either case, the user can choose how many songs they want to score at time for convenience. The relative case is an iterative process while the absolute case requires only one entry per feature per song. Once a dataset is ready to be cached, whether it's just names or more data, the user can name it, and it's saved to the data folder, with a note about which type of data it is.

The user has the option to use a cached dataset with just names and add data to it, or use only the names from a cached dataset with more data. Additionally, the user can use a cached dataset with absolute feature data and request to convert it to relative feature data and vice versa. None of these requests alter the cached data; conversions are stored temporarily or can be cached separately. 

After the user has configured the dataset they want to use for the tournament, they choose a playlist size (8 - size of dataset). If they don't choose to use the full dataset, they get a random subset with the specified size. The playlist is shuffled internally (not modifying the cached data) before the tournament runs. Lastly, the user must choose a tournament format (see '3. Tournament Formats' below) that is compatible with the dataset they have chosen, select a match size (number of songs they want to compare at a time) and then they run it. If relative or absolute feature data was provided, the user has the option to automate the tournament after the first round. A DQN will learn correlations between each feature and the user's match decisions in round 1 to call matches in the remaining rounds. If the user didn't use any data beyond song titles and artist labels, or just prefers to have complete control over the match winners, the can manually complete the tournament. 

The results of the tournament are printed to the console. The user can choose to see the full rankings, the top 5 ranked songs, or just the top song. If they provided feature data (absolute or relative), they will be able to see an analysis of how well each feature correlates to those rankings. 

**Phase 2: File Output** 
The only change made from phase 1 in this next phase is that the output will be a text or markdown file (.txt, .md, or .pdf). This gives the user something to keep from each run of the program (each of these files are stored in the results subfolder). The user can choose the desired format. 

**Phase 3: Python Executable File (.exe)** 
In Phase 3, I want to create a nice GUI and make it easier for the user to run the application. Instead of having to clone the repository and run driver code in Python, they can just download and run an executable file (.exe). When the file starts running, the user is prompted to select a cached playlist or import a new one. Then, they configure the data they want to use in the tournament (see 'Phase 1: Console Output' above). Next, the screen is divided into sections (buttons) that each represent a tournament format, and a textbox explaining them. After a selection is made, another dialog box appears to let the user choose a match size. 

Once the tournament is set up, the match window opens. The user is shown a match that they have to call, which feeds information to the program. Once the user completes all of the matches of the first round, they are prompted to automate the rest of the tournament or stick to manual decision-making. Then, when the tournament is finished, the results will be displayed on the screen. The user can select which subset of the rankings they want to see (one, top five, all). One section of the screen will have the ranked list, another will prominently show the winner of the tournament, and the last section will provide insights about how each feature contributed to the rankings (correlation between feature score and final score in the tournament). The user will be able to click a button and download the results in a format they choose (.txt, .md, .pdf). 

**Phase 4: Create Application With Spotify API** 
My ultimate vision for this project is to be able to run the application directly from Spotify or from a webpage that can access a Spotify playlist. This would make the input very clean and easy from the user's perspective (just click a button from the page of that playlist or import the playlist to an external site). Once the size of the playlist is verified to be compatible with the program, the application will launch in another tab or window, where they can run the tournament and get output as they can in phase 3. 

## 3. Tournament Formats
### a) Swiss (No Data Needed)
This format is the ideal option if the user doesn't provide additional song feature data, but is compatible with any data. Here is the general algorithm:

1. Sort the list of songs by current score (they all start with 0)
2. Group them into matches of the specified size by going down the list
3. Rank the songs in each match
4. Add to point totals (first adds match_size - 1, second adds one less, and so on)
5. Repeat steps 1-4 until one song has the sole highest score

One limitation of this method is that if song A beats song B in a match, song A is not guaranteed to have a better final ranking than song B. This may happen because real world preferences don't always follow the transitive property: if song A beats song B, and song B beats song C, then song A must beat song C. Depending on the outcomes of each match, it may not even be mathematically/logically possible to enforce a rule that guarantees that song A will have a higher final ranking than song B. 

### b) Sorting (Absolute Feature Data Recommended)
This format is the ideal option if the user provides absolute feature data, but is compatible with any or no data. Here is the general algorithm:

1. Assign each song a score of 0
2. Choose (match size - 1) songs at a random as a separators
3. Each match in the first round is the separators plus one other song, until every song has faced the separators
4. This process "sorts" the songs into pools based on how many of the separators they each beat
5. Notice that we automatically know the final rankings of the separators based on the size of each pool
6. Repeat steps 2-5 for each pool, until the pool size is less than or equal to the match size (in which case 1 match can rank that pool)
7. The tournament ends when there are no more pools with a size greater than 1

The user manually provides the results for each match in the first round, and if they provided data, that trains a DQN to analyze how each feature in the dataset contributed to those calls. Then, the DQN will complete the tournament automatically. The user can also finish the rest of the tournament themselves if they provided no data or wish to do it. To prevent the user from having to re-rank the separators in every match, save the rankings of the separators from the first match of a round and just ask the user where they would insert the new song. 

Unlike in Swiss tournament, if song A beats song B in any match in any round, it is guaranteed to have a higher ranking at the end because it will be moved up into a higher pool. However, one limitation of this format is that the DQN may not accurately reflect user preferences.

### c) Hybrid (Relative Feature Data Required)
Relative feature data is required for a hybrid format because any song that is strictly dominated by another song (has a strictly lower feature score for all features) is eliminated from the tournament and not displayed in the final rankings. If the user wants to see complete rankings of all songs, they should not use this method. This dropout could technically work with absolute feature data, but it would potentially eliminate most of the songs. The way relative feature data is constructed gives us a nice way to weed out strictly dominated songs without eliminating as many. After the dropout of strictly dominated songs, the user has the option to start a smaller Swiss or Sorting tournament with the songs that remain.

Since the user is in control of the relative data, they have more influence over the final result than in the Sorting model. However, the drawback of this method is the extra effort required to get the relative feature data (the other methods don't require data).

## 4. Design Considerations
If the chosen match size for the tournament does not divide the total number of songs, pad the list with dummy songs so the match size remains constant. The user should never see dummy songs, and dummy songs automatically get no points in a match so they don't affect the rankings. There should also be at most one dummy song in a match (for instance, if there are 13 songs and the match size is 3, there will be 3 matches of 3 real songs each, and 2 matches of 2 real songs and 1 dummy each).

## 5. Codebase Architecture
### Phase 1
```
FavoriteSongTournament/
├── data_loader.py            # Loads dataset from a filepath, user specifies if it is relative or absolute, methods to transform from one                                  to the other 
├── algorithms.py             # Stores logic for the tournament algorithms
├── results.py                # Displays the results of the tournament to the user (prints in console)
├── experiment.py             # Constructs the full experiment, including outputs
├── runner.py                 # Driver code, takes in user input, runs tournament
├── Kmeans_SongClusters.pdf   # Original mathematical problem specification
├── favorite_song_data.xlsx   # Original dataset, given as relative data, can be transformed to absolute, can remain unused if user wants a                                 Swiss tournament
├── tests/                    # Automated unit and integration test suite
│   ├── test_models.py        # Tests the logic of the tournaments
│   ├── test_math.py          # Ensure that all of the calculations are mathematically correct
│   ├── test_experiment.py    # Ensure that the pipeline runs correctly and outputs are in the correct format
│   └── test_smoke.py         # Rapid end-to-end integration smoke test
└── README.md                 # Project specification, experiment documentation, architecture, and usage
```

## 6. Installation & Usage
### Phase 1
#### Prerequisites
The project requires Python 3.8+ with the following packages:
- `pandas`
- `openpyxl`
- `numpy`
- `scipy`
- `torch`

Install dependencies via pip:
```bash
pip install pandas openpyxl numpy scipy torch
```

#### Running the Interactive Console Tournament
Run the interactive runner CLI:
```bash
python3 runner.py
```
You will be prompted to:
1. Enter your playlist size (between 8 and 128 songs).
2. Choose your data source (use the included `favorite_song_data.xlsx`, specify a custom file, or generate a synthetic playlist).
3. Select a tournament format:
   - **Swiss**: No initial data needed. Evaluates matches directly in score pools with Strength of Schedule tiebreakers.
   - **Sorting**: Multi-pivot quicksort algorithm. You rank matches manually in Round 1, training a Deep Q-Network (DQN) to automate subsequent rounds.
   - **Hybrid**: Analyzes relative criteria, filters out strictly dominated songs (songs strictly worse across all features), and runs a Swiss or Sorting tournament on surviving songs.
4. Select comparison criteria (default suggestions: Melody, Catchiness, Lyricism, Message, or custom criteria).
5. Choose match size (2 to 8 songs per match).
6. Rank songs in each match.
7. View your results: winner announcement, complete rankings table, and Pearson/Spearman feature correlations with music taste insights.

#### Automated Demo Mode
Run a quick, non-interactive demonstration:
```bash
# Swiss Tournament demo
python3 runner.py --demo --format swiss --size 16 --match-size 2

# Sorting Tournament demo with DQN automation
python3 runner.py --demo --format sorting --size 16 --match-size 3

# Hybrid Tournament demo with dominance filtering
python3 runner.py --demo --format hybrid --size 16 --match-size 2
```

#### Running Automated Tests
Run all unit, mathematical, pipeline, and integration smoke tests:
```bash
python3 -m unittest discover tests
```

