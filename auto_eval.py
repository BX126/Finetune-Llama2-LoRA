import re 
from process_data import *
import json
import math

import pickle
with open('/home/bingxuan/fine-tune/phenom.pkl', 'rb') as handle:
    phenom = pickle.load(handle)

def get_stress(phone):
    '''
    Takes in a piece of phonetics of a word, and produce the stress (1) and unstress (0).
    Example input: ['AH0 M EY1 Z IH0 NG']
    The phones could be retrived using the following command: phone = pronouncing.phones_for_word('amazing')
    Example output: [0, 1, 0]
    '''
    stress = []
    # print(phone)
    for s in phone.split():
        # print(s)
        if s[-1].isdigit():
            # print()
            if s[-1] == '2':
                stress.append(0)
            else:
                stress.append(int(s[-1]))
    # print("reached the end in get_stress")
    return stress

output= " The generated lyric is 'With every step, you take, I'm transported on a journey, feeling the rhythm, feeling alive', and the corresponding syllables for each word is With [W IH1 DH](/STRESSED/) every [EH1 V ER0 IY0](/STRESSED/-/UNSTRESSED/-/UNSTRESSED/) step, [S T EH1 P](/STRESSED/) you [Y UW1](/STRESSED/) take, [T EY1 K](/STRESSED/) I'm [AY1](/STRESSED/) transported [T R AE0 N S P AO1 R T IH0 D](/UNSTRESSED/-/STRESSED/-/UNSTRESSED/) on [AA1 N](/STRESSED/) a [AH0](/UNSTRESSED/) journey, [JH ER1 N IY0](/STRESSED/-/UNSTRESSED/) feeling [F IY1 L IH0 NG](/STRESSED/-/UNSTRESSED/) the [DH AH0](/UNSTRESSED/) rhythm, [R IH1 DH AH0 M](/STRESSED/-/UNSTRESSED/) feeling [F IY1 L IH0 NG](/STRESSED/-/UNSTRESSED/) alive [AH0 L AY1 V](/UNSTRESSED/-/STRESSED/). The number of syllables of the generated lyric matches with the total number of syllables in the music constraint. Generated lyric has 21 syllables. The important words in the generated lyric is ['step,', 'take,', 'transported', 'journey,', 'feeling', 'rhythm,', 'feeling']. The position of the stressed syllables of these important words are [4, 6, 11, 12, 15, 17, 18], and S_4, S_6, S_11, S_12, S_15, S_17, S_18 are all '/STRESSED/'. The position of stressed syllable of important words in the generated lyric matches the music constraint. Therefore, the generated lyric satisfies the music constraint. The generated lyric is coherent with the previously generated lyrics."
input_string = "Lyric that needed to be revised based on the music constraint: 'Each step you take takes me on a thrilling journey, filling the air with emotion and confidence'. Previously generated lyrics are: In a desert town where the sun shines bright, there's a funky flamingo who dances all night, With feathers so vibrant, it's a mesmerizing sight, The funky flamingo sets the stage alight, Filling the air with passion and pride, Its graceful movements, like a fiery ballet, Hips swaying freely, in a playful display, With every step, it captivates the crowd, The funky flamingo, dancing so proud, Oh, the Funky Flamingo Flamenco, see it soar, Dancing to the rhythm, it's begging for more., With every move, it takes you on a wild ride, filling the air with passion and pride, With every move, it takes you on a wild ride, filling the air with passion and pride, From dusk till dawn, it keeps the magic alive, spreading its charm and making the world come alive, The funky flamingo's rhythm has no end, A dance that transcends, making our souls mend., Underneath the moonlight, the night comes alive,, With the funky flamingo, it's a joyful vibe, Its wings spread wide, embracing the beat, The funky flamingo dances with so much heat. Title is 'Opera at the Laundromat'. The music constraint: S_0: /STRESSED/ S_1: /UNSTRESSED/ S_2: /UNSTRESSED/ S_3: /UNSTRESSED/ S_4: /STRESSED/ S_5: /UNSTRESSED/ S_6: /STRESSED/ S_7: /UNSTRESSED/ S_8: /UNSTRESSED/ S_9: /UNSTRESSED/ S_10: /UNSTRESSED/ S_11: /STRESSED/ S_12: /STRESSED/ S_13: /UNSTRESSED/ S_14: /UNSTRESSED/ S_15: /STRESSED/ S_16: /UNSTRESSED/ S_17: /STRESSED/ S_18: /UNSTRESSED/ S_19: /UNSTRESSED/ S_20: /STRESSED/. The goal is to firstly, match the number of syllables in the music constraint, and secondly, match the important word to the /STRESSED/ syllables. The music constraint indicates that there should be 21 syllables in the generated lyrics. The important words in the original lyric is ['step', 'take', 'takes', 'thrilling', 'journey,', 'filling', 'air', 'emotion', 'confidence'], and the syllables for each word is Each [IY1 CH](/STRESSED/) step [S T EH1 P](/STRESSED/) you [Y UW1](/STRESSED/) take [T EY1 K](/STRESSED/) takes [T EY1 K S](/STRESSED/) me [M IY1](/STRESSED/) on [AA1 N](/STRESSED/) a [AH0](/UNSTRESSED/) thrilling [TH R IH1 L IH0 NG](/STRESSED/-/UNSTRESSED/) journey, [JH ER1 N IY0](/STRESSED/-/UNSTRESSED/) filling [F IH1 L IH0 NG](/STRESSED/-/UNSTRESSED/) the [DH AH0](/UNSTRESSED/) air [EH1 R](/STRESSED/) with [W IH1 DH](/STRESSED/) emotion [IH0 M OW1 SH AH0 N](/UNSTRESSED/-/STRESSED/-/UNSTRESSED/) and [AH0 N D](/UNSTRESSED/) confidence [K AA1 N F AH0 D AH0 N S](/STRESSED/-/UNSTRESSED/-/UNSTRESSED/). The total number of syllables in original sentence is 24, and that does not match the number of syllables indicated by the music constraint. Therefore, we want to rephrase the sentence, so that 1, the number of syllables in the generated lyric is 21, 2, the stress of each of the important word in the generated lyric matches with the music constraint, and and 3, it is coherent with the previously generated lyrics"
def get_generated_lyrics(output):
    lyrics_match_single_quote = re.search("The generated lyric is '(.*?)', and", output)
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None

def get_original_lyrics(Input):
    lyrics_match_single_quote = re.search("Lyric that needed to be revised based on the music constraint: '(.*?)'. Previously", Input)
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None

def get_music_constraint(input_string):
    syllable_pattern = r"S_\d+: /STRESSED/|S_\d+: /UNSTRESSED/"
    # Extracting all syllable notations
    syllable_info = re.findall(syllable_pattern, input_string)
    stress_pattern_list = [1 if "/STRESSED/" in syllable else 0 for syllable in syllable_info]
    return stress_pattern_list
def get_title(Input):
    pattern = r"Title is '\s*(.*?)\s*'.\s*The music constraint"

    # Search for the pattern and extract the content
    match = re.search(pattern, Input, re.DOTALL) 
    return match.group(1) if match else None
common_word = []
#diff between the lyric and length of constraint
def syllable_match(lyrics,constraint):
    num_syllable = 0
    for word in lyrics.split():
        cleaned_word = remove_contractions(word)
        cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word).lower()
        phone = phenom[cleaned_word.strip().upper()]
        stress = get_stress(phone)
        num_syllable += len(stress)
        # try:
        #     cleaned_word = remove_contractions(word)
        #     cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word).lower()
        #     phone = pronouncing.phones_for_word(cleaned_word.strip())[0]
        #     stress = get_stress(phone)
        #     num_syllable += len(stress)
        # except Exception as e:
        #     print(e)
        #     cleaned_word = remove_contractions(word)
        #     cleaned_word = re.sub(r'[^\w\s\']', '', cleaned_word).lower()
        #     num_syllable +=len(get_uncommen_word_stress(cleaned_word))
    return num_syllable-len(constraint)

def get_uncommen_word_stress(word):
    while True:
        for i in common_word:
            if i["word"] == word:
                return i["stress"]
        q = f"Enter the stress of syllables in {word}: "
        input_string_str = input(q)
        try:
            stress = input_string_str.split()
            stress = [int(i) for i in stress]
            common_word.append({"word":word,"num_syllable":len(stress)})
            return stress
            break   
        except ValueError:
            print("That's not a valid list of int. Please enter a list of 1 or 0s.")
        # common_word.append({"word":word,"stress":stress})
        # return stress
    
def stress_match(lyrics,constraint):
    _,index = extract(lyrics)
    num_stress = len(index)
    incorrect = 0
    for i in index:
        if constraint[i] != 1:
            incorrect += 1
    return num_stress,incorrect

    # print("-----------")
    # print(song['res'])
    # print("-----------")
def get_constraint_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(46)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        constraint = get_music_constraint(Input)
        counter[len(constraint)] += 1
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Distribution of length of music constraint')
    plt.xticks(range(0, len(counter), 2))
    plt.savefig("dataset_music_constraint_distribution.png")
def get_diff_original_constraint_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(60)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        lyrics = get_original_lyrics(Input)
        cosntraint = get_music_constraint(Input)
        diff = syllable_match(lyrics,constraint)

        try:
            counter[diff+20] += 1
        except:
            print(lyrics)
            print(constraint)
            continue
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Difference between original sentence and len(music constraint)')
    plt.xlim(-20, 40)
    x_labels = [i - 20 for i in range(len(counter))]
    plt.xticks(range(0, len(counter), 2), labels=x_labels[::2])
    plt.savefig("dataset_diff_original_constraint_distribution.png")
def get_original_sentence_distribution(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    #check music_constraint len and number of syllables in the input
    counter = [0 for i in range(60)]
    # print(counter)
    for sentence in data:
        Input = sentence["Input"]
        lyrics = get_original_lyrics(Input)
        num_syllable,index = extract(lyrics)
        counter[num_syllable] += 1
    print(counter)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counter)), counter, color='skyblue')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Number of data points')
    plt.title('Distribution of number of syllales in sentences')
    plt.xticks(range(0, len(counter), 2))
    plt.savefig("dataset_original_sentence_distribution.png")
# get_constraint_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
# get_original_sentence_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
# get_diff_original_constraint_distribution("/home/songyan/Real_M2L-main/llama/generated_fine_tuned.json")
def data_analysis(file_path):
    with open("/home/songyan/Real_M2L-main/llama/generated_fine_tuned_result.json",'r') as file:
        data = json.load(file)
    total = 0
    miss = 0
    num_syllable_unmatch = []

    for song in data:
        lyrics = get_generated_lyrics(song["res"])
        if lyrics is None:
            print(song["res"])
            continue
        # print(lyrics)
        constraint = get_music_constraint(song["prompt"])
        # print(constraint)
        syllable_diff = syllable_match(lyrics,constraint) 
        if syllable_diff != 0:
                num_syllable_unmatch.append(abs(syllable_diff))
        # if syllable_diff >4:
        #     print(lyrics)
        #     print(constraint)
        #generated syllables <= number of notes
        if syllable_diff <= 0:
            cur_num_stress,cur_miss=stress_match(lyrics,constraint)
            total += cur_num_stress
            miss += cur_miss
        else:
            continue
    print(f'among lyrics that matches the number of syllables in music constraint: total_important_syllables {total},missed matched {1}')
    print(f'there are {len(num_syllable_unmatch)} lyrics that does not matches the number of syllables in music constraint. difference are {num_syllable_unmatch}')
    l = [0 for i in range(30)]
    for i in num_syllable_unmatch:
        l[i] += 1
    print(l)
# with open("/home/songyan/Real_M2L-main/llama/generated_fine_tuned_result.json",'r') as f:
#     data = json.load(f)
# i = 0
# prompt = data[2]["prompt"]
# generated_lyrics = get_generated_lyrics(data[2]["res"])
# music_constraint = get_music_constraint(data[2]["prompt"])
# diff = syllable_match(generated_lyrics,music_constraint)
# if diff == 0 or i >= 3:
#     finish = True
# title = get_title(prompt)
# history = get_previous_lyrics(prompt)
# Input = create_input(generated_lyrics,music_constraint,history,title)
# if Input == None:
#     finish = True
#     invalid += 1
# print(Input)

