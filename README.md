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

**Phase 2: File Output** 

**Phase 3: Python Executable File (.exe)** 

**Phase 4: Create Application With Spotify API** 

## 3. Tournament Formats
### a) Swiss (No Data Needed)
### b) Sorting (Absolute Feature Data Required)
### c) Hybrid (Relative Feature Data Required)
## 4. Codebase Architecture
## 5. Installation & Usage
